from .base_task import BaseTask
from typing import List
from environment.parameters import *
from isaacsim.core.utils.prims import is_prim_path_valid,get_prim_at_path, get_all_matching_child_prims
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.core.utils.types import ArticulationAction

import omni
import torch
from isaacsim.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from local_utils.env import position_reached, rotation_reached, get_pre_grasp_action

import isaaclab.sim as sim_utils

import logging
import numpy as np
import json as _json
import time as _time

# #region agent log
def _dbg(path, **kw):
    kw.setdefault("timestamp", int(_time.time()*1000))
    kw.setdefault("sessionId", "2ed6cc")
    with open(path, "a") as _f:
        _f.write(_json.dumps(kw) + "\n")
# #endregion

class PickupObject(BaseTask):
    # #region agent log
    _DBG_LOG = "/home/rgong/Desktop/arnold/.cursor/debug-2ed6cc.log"
    # #endregion

    def __init__(self, num_stages, horizon, stage_properties, record) -> None:
        super().__init__(num_stages, horizon, stage_properties, record)
        self.task = 'pickup_object'
        self.gripper_trigger_period = 50
        self.success_check_period = 300
        self.grip_open = [True, False, False]
        self.logger = logging.getLogger(__name__)
        self.use_gpu_physics = False

    def reset(self, robot_parameters, 
              scene_parameters, 
              object_parameters,
              robot_base,
              gt_actions
        ):

        print('[PickupObject.reset] calling super().stop()', flush=True)
        super().stop()
        print('[PickupObject.reset] super().stop() done', flush=True)

        self.robot_parameters: RobotParameters = robot_parameters
        self.object_parameter: ObjectParameters = object_parameters[0]
        self.stage = omni.usd.get_context().get_stage()
        self.checker = None

        self.robot_base = robot_base

        print(f'[PickupObject.reset] calling super().reset() _robot_loaded={self._robot_loaded}', flush=True)
        obs = super().reset(
            robot_parameters = robot_parameters,
            scene_parameters = scene_parameters
        )
        print('[PickupObject.reset] super().reset() done', flush=True)
        self.current_stage = 0
        self.end_stage = 0

        self.time_step = 0
        self.is_success = 0
        self.gt_actions = gt_actions

        return obs

    def set_up_task(self) -> None:
        self.load_object()

    def load_object(self):
        self.objects_list = []
        param = self.object_parameter

        object_prim_path, object_prim = self._prepare_object_prim(0, param.usd_path)

        self._wait_for_loading()

        self.objects_list.append(object_prim)

        positions = torch.tensor(np.array(param.object_position)/100.0).unsqueeze(0)
        rotations = torch.tensor(param.orientation_quat).unsqueeze(0)
        scales = torch.tensor(np.array(param.scale)/100.0).unsqueeze(0)

        # #region agent log
        _dbg(self._DBG_LOG, hypothesisId="H", location="pickup_object.py:load_object",
             message="object loaded",
             data={"prim_path": object_prim_path,
                   "raw_position_cm": np.array(param.object_position).tolist(),
                   "position_m": positions.squeeze(0).tolist(),
                   "scale_m": scales.squeeze(0).tolist()})
        # #endregion

        # use this to set relative position, orientation and scale
        XFormPrim(object_prim_path, positions= positions, orientations = rotations, scales = scales)
        self._wait_for_loading()

        if param.object_physics_properties:
            set_physics_properties(self.stage, object_prim, param.object_physics_properties)
            
        add_update_semantics(object_prim, param.object_type)

        if param.object_timeline_management is not None:
            self.checker = param.object_timeline_management
            self.checker.pre_initialize(object_prim_path)

    def step(self, act_pos, act_rot, render, use_gt):
        """
        `act_pos`: np.ndarray (3,)
        `act_rot`: np.ndarray (4,) (wxyz)
        `render`: bool
        `use_gt`: bool
        `step` is called twice, first for grasping object and second for manipulating object
        """
        simulation_context = sim_utils.SimulationContext.instance()
        current_target = None

        if self.current_stage == 0:
            self.end_stage = 2
            if use_gt:
                self.trans_pick, self.rotat_pick = self.gt_actions[1]
                self.trans_pick = np.array(self.trans_pick)/100.0
            else:
                self.trans_pick = act_pos
                self.rotat_pick = act_rot

            # #region agent log
            _pre_raw = np.array(self.gt_actions[0][0]) if use_gt else None
            _dbg(self._DBG_LOG, hypothesisId="H", location="pickup_object.py:step_gt_actions",
                 message="gt_actions used in step",
                 data={"step_call": 1,
                       "pre_grasp_cm": _pre_raw.tolist() if _pre_raw is not None else None,
                       "grasp_cm": np.array(self.gt_actions[1][0]).tolist() if use_gt else None,
                       "lift_cm": np.array(self.gt_actions[2][0]).tolist() if use_gt and self.gt_actions[2] is not None else None,
                       "grasp_m": self.trans_pick.tolist()})
            # #endregion
        else:
            self.end_stage = self.num_stages
            if use_gt:
                self.trans_target, self.rotat_target = self.gt_actions[2]
                self.trans_target = np.array(self.trans_target)/100.0
            else:
                self.trans_target = act_pos
                self.rotat_target = act_rot

            # #region agent log
            _dbg(self._DBG_LOG, hypothesisId="H", location="pickup_object.py:step_gt_actions",
                 message="gt_actions used in step (lift)",
                 data={"step_call": 2,
                       "lift_cm": np.array(self.gt_actions[2][0]).tolist() if use_gt else None,
                       "lift_m": self.trans_target.tolist()})
            # #endregion

        # #region agent log
        _robot_pos, _robot_rot = self.robot.get_world_pose()
        _ee_pos_init = self._get_ee_pos()
        _ctrl_base_pos = self.c_controller._default_position
        _ctrl_base_rot = self.c_controller._default_orientation
        _jpos_init = self.robot.get_joint_positions()
        _dbg(self._DBG_LOG, hypothesisId="A,C,E", location="pickup_object.py:step_entry",
             message="step entry state",
             data={"current_stage": self.current_stage, "end_stage": self.end_stage,
                   "robot_world_pos": _robot_pos.tolist() if hasattr(_robot_pos,'tolist') else list(_robot_pos),
                   "robot_world_rot": _robot_rot.tolist() if hasattr(_robot_rot,'tolist') else list(_robot_rot),
                   "ctrl_base_pos": _ctrl_base_pos.tolist() if hasattr(_ctrl_base_pos,'tolist') else list(_ctrl_base_pos),
                   "ctrl_base_rot": _ctrl_base_rot.tolist() if hasattr(_ctrl_base_rot,'tolist') else list(_ctrl_base_rot),
                   "ee_pos_init": _ee_pos_init.tolist(),
                   "joint_positions_init": _jpos_init.tolist() if hasattr(_jpos_init,'tolist') else list(_jpos_init),
                   "time_step": self.time_step})
        # #endregion
        
        stage_step = 0
        stall = {"last_pos": None, "stall_count": 0}
        _last_arm_cmd = None
        _gripper_open = (self.current_stage == 0)

        while self.current_stage < self.end_stage:
            if self.time_step % 120 == 0:
                self.logger.info(f"tick: {self.time_step}")
            
            if self.time_step >= self.horizon:
                self.is_success = -1
                break

            if self._check_stall(stall, stage_step):
                print(f'[pickup] stage {self.current_stage} stalled, skipping',
                      flush=True)
                self.is_success = -1
                break

            if current_target is None:
                grip_open = self.grip_open[self.current_stage]

                if self.current_stage == 0:
                    if use_gt:
                        trans_pre, rotation_pre = self.gt_actions[0]
                        trans_pre = np.array(trans_pre)/100.0
                    else:
                        trans_pre, rotation_pre = get_pre_grasp_action(
                            grasp_action=(self.trans_pick, self.rotat_pick),
                            robot_base=self.robot_base, task=self.task
                        )
                    current_target = (trans_pre, rotation_pre, grip_open)

                elif self.current_stage == 1:
                    current_target = (self.trans_pick, self.rotat_pick, grip_open)
                
                else:
                    current_target = (self.trans_target, self.rotat_target, grip_open)

                # #region agent log
                _dbg(self._DBG_LOG, hypothesisId="C", location="pickup_object.py:new_target",
                     message="new stage target set",
                     data={"stage": self.current_stage,
                           "target_pos": np.array(current_target[0]).tolist(),
                           "target_rot": np.array(current_target[1]).tolist(),
                           "grip_open": current_target[2],
                           "ee_pos_now": self._get_ee_pos().tolist(),
                           "dist_to_target": float(np.linalg.norm(self._get_ee_pos() - np.array(current_target[0])))})
                # #endregion
            
            if position_reached( self.c_controller, current_target[0], self.robot, thres=(0.002 if self.current_stage == 1 else 0.005) ) \
            and rotation_reached( self.c_controller, current_target[1] ):
                # #region agent log
                _dbg(self._DBG_LOG, hypothesisId="D", location="pickup_object.py:target_reached",
                     message="target reached - advancing stage",
                     data={"stage": self.current_stage, "time_step": self.time_step, "stage_step": stage_step,
                           "ee_pos": self._get_ee_pos().tolist(),
                           "target_pos": np.array(current_target[0]).tolist()})
                # #endregion

                joint_positions = self.robot.get_joint_positions()
                gripper_state = joint_positions[-2:]
                current_gripper_open = (gripper_state[0] + gripper_state[1] > 0.07)

                if current_target[2] != current_gripper_open:
                    num_dofs = self.robot.num_dof
                    gripper_indices = [num_dofs - 2, num_dofs - 1]  # panda_finger_joint1, panda_finger_joint2
                    if current_target[2] < 0.5:
                        gripper_positions = np.array([0.0, 0.0])  # close
                    else:
                        gripper_positions = np.array([0.05, 0.05])  # open
                    target_joint_positions_gripper = ArticulationAction(
                        joint_positions=gripper_positions, joint_indices=gripper_indices
                    )
                    _gripper_open = current_target[2]
                    for _ in range(self.gripper_trigger_period):
                        if _last_arm_cmd is not None:
                            _arm_idx = torch.arange(7)
                            _arm_pos = torch.tensor(np.array(_last_arm_cmd[:7], dtype=np.float32)).unsqueeze(0)
                            self.robot.set_joint_positions(_arm_pos, joint_indices=_arm_idx)
                            self.robot.set_joint_velocities(torch.zeros_like(_arm_pos), joint_indices=_arm_idx)
                        articulation_controller = self.robot.get_articulation_controller()
                        articulation_controller.apply_action(target_joint_positions_gripper)
                        self.try_record(actions=target_joint_positions_gripper)
                        simulation_context.step(render=render)

                current_target = None
                self.current_stage += 1
                stage_step = 0
                stall = {"last_pos": None, "stall_count": 0}
                self.logger.info(f"enter stage {self.current_stage}")

                # #region agent log
                if self.checker is not None and hasattr(self.checker, 'targetRigid'):
                    _obj_pos, _ = self.checker.targetRigid.get_world_poses(usd=False)
                    _obj_y = _obj_pos[0][1].item()
                    _dbg(self._DBG_LOG, hypothesisId="Q,R", location="pickup_object.py:stage_transition",
                         message="object Y after stage transition",
                         data={"new_stage": self.current_stage, "object_y": _obj_y,
                               "target_prim_init_y": self.checker.target_prim_init_y if hasattr(self.checker,'target_prim_init_y') else None,
                               "target_height": (self.checker.target_delta_y + self.checker.target_prim_init_y) if hasattr(self.checker,'target_prim_init_y') else None,
                               "ee_pos_y": self._get_ee_pos()[1]})
                # #endregion
            
            else:
                
                target_joint_positions = self.c_controller.forward(
                    target_end_effector_position=current_target[0], target_end_effector_orientation=current_target[1]
                )

                # #region agent log
                if stage_step % 60 == 0:
                    _ee_now = self._get_ee_pos()
                    _dist = float(np.linalg.norm(_ee_now - np.array(current_target[0])))
                    _jt = target_joint_positions.joint_positions
                    _jcur = self.robot.get_joint_positions()
                    _ji = target_joint_positions.joint_indices
                    _dbg(self._DBG_LOG, hypothesisId="B,E", location="pickup_object.py:forward_result",
                         message="controller forward output",
                         data={"stage": self.current_stage, "time_step": self.time_step,
                               "stage_step": stage_step,
                               "ee_pos": _ee_now.tolist(),
                               "target_pos": np.array(current_target[0]).tolist(),
                               "dist_to_target": _dist,
                               "commanded_joints": _jt.tolist() if hasattr(_jt, 'tolist') else (list(_jt) if _jt is not None else None),
                               "current_joints": _jcur.tolist() if hasattr(_jcur, 'tolist') else list(_jcur),
                               "action_joint_indices": _ji.tolist() if hasattr(_ji, 'tolist') else (list(_ji) if _ji is not None else None),
                               "gripper_open_flag": _gripper_open})
                # #endregion
                
                if target_joint_positions.joint_positions is not None:
                    _cmd = target_joint_positions.joint_positions
                    _last_arm_cmd = _cmd
                    _arm_idx = torch.arange(7)
                    _arm_pos = torch.tensor(np.array(_cmd[:7], dtype=np.float32)).unsqueeze(0)
                    self.robot.set_joint_positions(_arm_pos, joint_indices=_arm_idx)
                    self.robot.set_joint_velocities(torch.zeros_like(_arm_pos), joint_indices=_arm_idx)

                articulation_controller = self.robot.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions)
                _grip_pos = np.array([0.04, 0.04]) if _gripper_open else np.array([0.0, 0.0])
                _grip_idx = [self.robot.num_dof - 2, self.robot.num_dof - 1]
                articulation_controller.apply_action(
                    ArticulationAction(joint_positions=_grip_pos, joint_indices=_grip_idx)
                )
                self.try_record(actions=target_joint_positions)

            simulation_context.step(render=render)

            # #region agent log
            if stage_step < 3 or stage_step % 20 == 0:
                _jp_post = self.robot.get_joint_positions()
                _dbg(self._DBG_LOG, hypothesisId="GR1,GR2,GR3", location="pickup_object.py:post_step",
                     message="joints after step (gripper PD fix active)",
                     data={"stage": self.current_stage, "time_step": self.time_step,
                           "stage_step": stage_step,
                           "gripper_joints": [float(_jp_post[-2]), float(_jp_post[-1])],
                           "gripper_open_flag": _gripper_open,
                           "arm_j0_j6": [float(_jp_post[0]), float(_jp_post[6])],
                           "last_arm_cmd": [float(x) for x in _last_arm_cmd[:7]] if _last_arm_cmd is not None else None})
            # #endregion

            self.time_step += 1
            stage_step += 1

        # #region agent log
        _scp_obj_y_start = None
        if self.checker is not None and hasattr(self.checker, 'targetRigid'):
            _sp, _ = self.checker.targetRigid.get_world_poses(usd=False)
            _scp_obj_y_start = _sp[0][1].item()
        _dbg(self._DBG_LOG, hypothesisId="Q,R,S", location="pickup_object.py:success_check_start",
             message="entering success_check_period",
             data={"is_success_before": self.is_success, "object_y": _scp_obj_y_start,
                   "success_steps": self.checker.success_steps if self.checker else None,
                   "target_height": (self.checker.target_delta_y + self.checker.target_prim_init_y) if (self.checker and hasattr(self.checker,'target_prim_init_y')) else None,
                   "current_stage": self.current_stage, "end_stage": self.end_stage})
        # #endregion

        for _scp_i in range(self.success_check_period):
            simulation_context.step(render=False)
            # #region agent log
            if _scp_i % 50 == 0 and self.checker is not None and hasattr(self.checker, 'targetRigid'):
                _sp2, _ = self.checker.targetRigid.get_world_poses(usd=False)
                _scp_obj_y = _sp2[0][1].item()
                _th = self.checker.target_delta_y + self.checker.target_prim_init_y
                _dbg(self._DBG_LOG, hypothesisId="Q,R,S", location="pickup_object.py:success_check_step",
                     message="success_check_period step",
                     data={"scp_step": _scp_i, "object_y": _scp_obj_y,
                           "need_delta_y": abs(_scp_obj_y - _th),
                           "vel": self.checker.vel, "success_steps": self.checker.success_steps,
                           "checker_success": self.checker.success})
            # #endregion
            if self.checker.success:
                self.is_success = 1
                break

        # #region agent log
        _scp_obj_y_end = None
        if self.checker is not None and hasattr(self.checker, 'targetRigid'):
            _sp3, _ = self.checker.targetRigid.get_world_poses(usd=False)
            _scp_obj_y_end = _sp3[0][1].item()
        _dbg(self._DBG_LOG, hypothesisId="Q,R,S", location="pickup_object.py:success_check_end",
             message="success_check_period DONE",
             data={"is_success": self.is_success, "object_y_end": _scp_obj_y_end,
                   "object_y_start": _scp_obj_y_start,
                   "success_steps": self.checker.success_steps if self.checker else None,
                   "checker_success": self.checker.success if self.checker else None,
                   "target_height": (self.checker.target_delta_y + self.checker.target_prim_init_y) if (self.checker and hasattr(self.checker,'target_prim_init_y')) else None})
        # #endregion
        
        return self.render(), self.is_success
