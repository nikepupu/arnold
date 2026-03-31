# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation
import json as _json, time as _time

# #region agent log
_DBG_LOG_CTRL = "/home/rgong/Desktop/arnold/.cursor/debug-2ed6cc.log"
def _dbg_ctrl(**kw):
    kw.setdefault("timestamp", int(_time.time()*1000))
    kw.setdefault("sessionId", "2ed6cc")
    with open(_DBG_LOG_CTRL, "a") as _f:
        _f.write(_json.dumps(kw) + "\n")
# #endregion


class RMPFlowController(mg.MotionPolicyController):
    """[summary]

    Args:
        name (str): [description]
        robot_articulation (SingleArticulation): [description]
        physics_dt (float, optional): [description]. Defaults to 1.0/60.0.
    """

    def __init__(self, name: str, robot_articulation: SingleArticulation, physics_dt: float = 1.0 / 60.0) -> None:
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.rmp_flow.set_ignore_state_updates(True)

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmp_flow, physics_dt)

        # import ipdb; ipdb.set_trace()
        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._default_position = self._default_position.cpu().numpy()
        self._default_orientation = self._default_orientation.cpu().numpy()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        # #region agent log
        _jpos = self._articulation_motion_policy.get_active_joints_subset().get_joint_positions()
        _ee_pos, _ee_rot_mat = self._motion_policy.get_end_effector_pose(_jpos)
        _dbg_ctrl(hypothesisId="A", location="controller.py:__init__",
                  message="RMPFlowController initialized",
                  data={"default_position": self._default_position.tolist(),
                        "default_orientation": self._default_orientation.tolist(),
                        "initial_joint_positions": _jpos.tolist() if hasattr(_jpos,'tolist') else list(_jpos),
                        "computed_ee_pos": _ee_pos.tolist() if hasattr(_ee_pos,'tolist') else list(_ee_pos)})
        # #endregion
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
