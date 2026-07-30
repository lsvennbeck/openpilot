"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import CAR_ROTATION_RADIUS
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.models.helpers import plan_x_idxs_helper

TRAJECTORY_SIZE = ModelConstants.IDX_N
CAMERA_OFFSET = 0.04  # meters from center car to camera, TICI
PATH_OFFSET = 0.00

LAT_MPC_N = 32  # matches lat_mpc.py's compiled horizon (N=32, i.e. TRAJECTORY_SIZE-1 points)


class LanePlanner:
  """
  Blends a laneline-derived center path into the model's own predicted path,
  ported from openpilot v0.8.14's selfdrive/controls/lib/lane_planner.py (the last
  version where laneline-based lane centering was real/default), adapted to read
  the raw parsed model output dict instead of the published modelV2 cereal message.
  """

  def __init__(self):
    self.ll_t = np.zeros((TRAJECTORY_SIZE,))
    self.ll_x = np.zeros((TRAJECTORY_SIZE,))
    self.lll_y = np.zeros((TRAJECTORY_SIZE,))
    self.rll_y = np.zeros((TRAJECTORY_SIZE,))
    self.lane_width_estimate = FirstOrderFilter(3.7, 9.95, DT_MDL)
    self.lane_width_certainty = FirstOrderFilter(1.0, 0.95, DT_MDL)
    self.lane_width = 3.7

    self.lll_prob = 0.
    self.rll_prob = 0.
    self.d_prob = 0.

    self.lll_std = 0.
    self.rll_std = 0.

    self.camera_offset = CAMERA_OFFSET
    self.path_offset = PATH_OFFSET

  def parse_model(self, model_output: dict[str, np.ndarray]) -> None:
    lane_lines = model_output.get('lane_lines')
    lane_lines_prob = model_output.get('lane_lines_prob')
    lane_lines_stds = model_output.get('lane_lines_stds')
    if lane_lines is None or lane_lines_prob is None or lane_lines_stds is None:
      return
    if lane_lines.shape[1] != 4 or lane_lines.shape[2] != TRAJECTORY_SIZE:
      return

    self.ll_t = np.array(plan_x_idxs_helper(ModelConstants, Plan, model_output))
    self.ll_x = np.array(ModelConstants.X_IDXS)
    # index 1 = left_near, index 2 = right_near (the lines adjacent to the ego lane)
    self.lll_y = lane_lines[0, 1, :, 0] + self.camera_offset
    self.rll_y = lane_lines[0, 2, :, 0] + self.camera_offset
    probs = lane_lines_prob[0, 1::2]
    self.lll_prob = float(probs[1])
    self.rll_prob = float(probs[2])
    self.lll_std = float(lane_lines_stds[0, 1, 0, 0])
    self.rll_std = float(lane_lines_stds[0, 2, 0, 0])

  def get_d_path(self, v_ego: float, path_t: np.ndarray, path_xyz: np.ndarray) -> np.ndarray:
    path_xyz = path_xyz.copy()
    path_xyz[:, 1] += self.path_offset
    l_prob, r_prob = self.lll_prob, self.rll_prob

    # Reduce reliance on lanelines that are too far apart or will be in a few seconds
    width_pts = self.rll_y - self.lll_y
    prob_mods = []
    for t_check in (0.0, 1.5, 3.0):
      width_at_t = np.interp(t_check * (v_ego + 7), self.ll_x, width_pts)
      prob_mods.append(np.interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
    mod = min(prob_mods)
    l_prob *= mod
    r_prob *= mod

    # Reduce reliance on uncertain lanelines
    l_std_mod = np.interp(self.lll_std, [.15, .3], [1.0, 0.0])
    r_std_mod = np.interp(self.rll_std, [.15, .3], [1.0, 0.0])
    l_prob *= l_std_mod
    r_prob *= r_std_mod

    # Find current lanewidth
    self.lane_width_certainty.update(l_prob * r_prob)
    current_lane_width = abs(self.rll_y[0] - self.lll_y[0])
    self.lane_width_estimate.update(current_lane_width)
    speed_lane_width = np.interp(v_ego, [0., 31.], [2.8, 3.5])
    self.lane_width = self.lane_width_certainty.x * self.lane_width_estimate.x + \
      (1 - self.lane_width_certainty.x) * speed_lane_width

    clipped_lane_width = min(4.0, self.lane_width)
    path_from_left_lane = self.lll_y + clipped_lane_width / 2.0
    path_from_right_lane = self.rll_y - clipped_lane_width / 2.0

    self.d_prob = l_prob + r_prob - l_prob * r_prob
    lane_path_y = (l_prob * path_from_left_lane + r_prob * path_from_right_lane) / (l_prob + r_prob + 0.0001)
    safe_idxs = np.isfinite(self.ll_t)
    if safe_idxs[0]:
      lane_path_y_interp = np.interp(path_t, self.ll_t[safe_idxs], lane_path_y[safe_idxs])
      path_xyz[:, 1] = self.d_prob * lane_path_y_interp + (1.0 - self.d_prob) * path_xyz[:, 1]
    else:
      cloudlog.warning("laneful mode: NaNs in laneline times, ignoring")
    return path_xyz


class LanefulCurvature:
  """
  Stateful helper computing curvature from a laneline-blended path via LateralMpc,
  reviving the pre-2022-removal control approach for models that predate openpilot's
  switch to pure model-path-following (see openpilot v0.8.14's
  selfdrive/controls/lib/lateral_planner.py). Heading and yaw-rate targets always come
  straight from the model's own prediction -- only the lateral (Y) position target is
  ever blended with lanelines, matching the original design.
  """

  def __init__(self):
    from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
    self.LP = LanePlanner()
    self.lat_mpc = LateralMpc()
    self.x0 = np.zeros(4)
    self.solution_invalid_cnt = 0

  def get_curvature(self, model_output: dict[str, np.ndarray], curvature_plan: np.ndarray,
                    v_ego: float, lat_action_t: float, prev_curvature: float) -> float:
    self.LP.parse_model(model_output)

    path_xyz = curvature_plan[:, Plan.POSITION]
    t_idxs = np.array(ModelConstants.T_IDXS)
    d_path_xyz = self.LP.get_d_path(v_ego, t_idxs, path_xyz)

    heading_pts = curvature_plan[:, Plan.T_FROM_CURRENT_EULER][:, 2]
    yaw_rate_pts = curvature_plan[:, Plan.ORIENTATION_RATE][:, 2]

    self.lat_mpc.set_weights(1., .1, 0.0, .05, 800)
    p = np.column_stack([v_ego * np.ones(LAT_MPC_N + 1), np.full(LAT_MPC_N + 1, CAR_ROTATION_RADIUS)])

    y_pts = d_path_xyz[:LAT_MPC_N + 1, 1]

    self.x0[3] = prev_curvature
    self.lat_mpc.run(self.x0, p, y_pts, heading_pts[:LAT_MPC_N + 1], yaw_rate_pts[:LAT_MPC_N + 1])

    mpc_nans = np.isnan(self.lat_mpc.x_sol[:, 3]).any()
    if mpc_nans or self.lat_mpc.solution_status != 0:
      self.lat_mpc.reset(x0=self.x0)
      cloudlog.warning("laneful mode: lateral mpc nan/invalid solution, resetting")
      return prev_curvature

    self.solution_invalid_cnt = self.solution_invalid_cnt + 1 if self.lat_mpc.cost > 20000. else 0

    # warm-start next call's x0 at one model timestep ahead, matching the original design
    self.x0[3] = np.interp(t_idxs[1], t_idxs[:LAT_MPC_N + 1], self.lat_mpc.x_sol[:, 3])
    # report the curvature at the actuator-delay-compensated horizon, consistent with
    # how get_curvature_from_plan() evaluates the non-laneful path at the same lat_action_t
    return float(np.interp(lat_action_t, t_idxs[:LAT_MPC_N + 1], self.lat_mpc.x_sol[:, 3]))
