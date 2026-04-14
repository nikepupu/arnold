from .base_task import BaseTask
from typing import List
from environment.parameters import *
from isaacsim.core.utils.prims import is_prim_path_valid,get_prim_at_path, get_all_matching_child_prims
from renewed_utils.semantics import add_update_semantics
from isaacsim.core.utils.types import ArticulationAction

import omni
import torch
from isaacsim.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from local_utils.env import position_reached, rotation_reached, get_pre_grasp_action

import isaaclab.sim as sim_utils

import logging
import numpy as np

# from renewed_utils.draw import draw_ee_target

class PickupObject(BaseTask):

    def __init__(self, num_stages, horizon, stage_properties, record, orient_patch = False) -> None:
        super().__init__(num_stages, horizon, stage_properties, record)
        self.task = 'pickup_object'
        self.gripper_trigger_period = 50
        self.success_check_period = 300
        self.grip_open = [True, False, False]
        self.logger = logging.getLogger(__name__)
        self.use_gpu_physics = False
        self.orient_patch = orient_patch

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
        scales = torch.ones_like(positions) / 10
        #self._bake_object_scale(object_prim_path, np.array(param.scale) / 100.0)

        XFormPrim(object_prim_path, positions=positions, orientations=rotations, scales=scales)
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

        else:
            self.end_stage = self.num_stages
            if use_gt:
                self.trans_target, self.rotat_target = self.gt_actions[2]
                self.trans_target = np.array(self.trans_target)/100.0

                if self.orient_patch:
                    from renewed_patches.orient import reset_target
                    has_patch, target_y_angle = reset_target(self.file_name, self.split)

                    if has_patch:
                        self.checker.target_delta_y = target_y_angle

                # import ipdb; ipdb.set_trace()
            else:
                self.trans_target = act_pos
                self.rotat_target = act_rot

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
                print(f'[pickup] stage {self.current_stage} stalled, advancing',
                      flush=True)
                current_target = None
                self.current_stage += 1
                stage_step = 0
                stall = {"last_pos": None, "stall_count": 0}
                continue

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

            if position_reached( self.c_controller, current_target[0], self.robot, thres=(0.001 if self.current_stage == 1 else 0.005) ) \
            and rotation_reached( self.c_controller, current_target[1] ):

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
                    
                    for _ in range(self.gripper_trigger_period):
                        articulation_controller = self.robot.get_articulation_controller()
                        articulation_controller.apply_action(target_joint_positions_gripper)
                        self.try_record(actions=target_joint_positions_gripper)
                        simulation_context.step(render=render)

                current_target = None
                self.current_stage += 1
                stage_step = 0
                stall = {"last_pos": None, "stall_count": 0}
                self.logger.info(f"enter stage {self.current_stage}")

            else:
                # # debug draw
                # draw_ee_target(current_target[0], current_target[1])

                target_joint_positions = self.c_controller.forward(
                    target_end_effector_position=current_target[0], target_end_effector_orientation=current_target[1]
                )

                articulation_controller = self.robot.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions)
                self.try_record(actions=target_joint_positions)

            simulation_context.step(render=render)

            self.time_step += 1
            stage_step += 1

        for _scp_i in range(self.success_check_period):
            simulation_context.step(render=False)
            if self.checker.success:
                self.is_success = 1
                break

        return self.render(), self.is_success
