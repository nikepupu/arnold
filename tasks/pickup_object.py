from .base_task import BaseTask
from typing import List
from environment.parameters import *
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.semantics import add_update_semantics
from utils.env import position_reached, rotation_reached, get_pre_grasp_action
from omni.isaac.core.simulation_context import SimulationContext
import omni
from omni.isaac.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
import logging
import os


class PickupObject(BaseTask):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)
        self.task = 'pickup_object'
        self.grip_open = cfg.gripper_open[self.task]
        self.logger = logging.getLogger(__name__)
        self.use_gpu_physics = False

    def reset(self, robot_parameters, 
              scene_parameters, 
              object_parameters,
              robot_base,
              gt_actions
        ):

        super().stop()

        self.robot_parameters: RobotParameters = robot_parameters
        self.object_parameter: ObjectParameters = object_parameters[0]
        self.stage = omni.usd.get_context().get_stage()
        self.checker = None

        self.robot_base = robot_base

        obs = super().reset(
            robot_parameters = robot_parameters,
            scene_parameters = scene_parameters
        )
        # used for max mumber of steps (grasp, raise up)
        self.current_stage = 0
        self.end_stage = 0

        self.time_step = 0
        self.is_success = 0
        self.gt_actions = gt_actions

        return obs

    def set_up_task(self) -> None:
        self.load_object()

    

    def load_object(self):
        # TODO:
        # For now only supports one environment, we will use cloner in the future
        index = 0
        self.objects_list = []
        param = self.object_parameter
        
        object_prim_path = find_unique_string_name(
            initial_name = f"/World_{index}/{param.object_type}",
            is_unique_fn = lambda x: not is_prim_path_valid(x)
        )

        self.usd_path = param.usd_path
        self.object_id = self.usd_path.split(os.sep)[-2]
        object_prim = add_reference_to_stage(param.usd_path, object_prim_path)
    
        self._wait_for_loading()

        self.objects_list.append(object_prim)

        position = param.object_position
        rotation = param.orientation_quat
    
        object_pos, object_rot = self._y_up_to_z_up(position=position, rotation=rotation)
        
        # use this to set relative position, orientation and scale
        xform_prim = XFormPrim(object_prim_path, translation= object_pos, orientation = object_rot, scale = np.array(param.scale))
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
        simulation_context = SimulationContext.instance()
        current_target = None

        if self.current_stage == 0:
            self.end_stage = 2
            if use_gt:
                self.trans_pick, self.rotat_pick = self.gt_actions[1]
                self.trans_pick, self.rotat_pick = self._y_up_to_z_up(position=self.trans_pick, rotation=self.rotat_pick)
            else:
                self.trans_pick = act_pos
                self.rotat_pick = act_rot
        else:
            self.end_stage = self.num_stages
            if use_gt:
                self.trans_target, self.rotat_target = self.gt_actions[2]
                self.trans_target, self.rotat_target = self._y_up_to_z_up(position=self.trans_target, rotation=self.rotat_target)
            else:
                self.trans_target = act_pos
                self.rotat_target = act_rot
        
        while self.current_stage < self.end_stage:
            if self.time_step % 120 == 0:
                self.logger.info(f"tick: {self.time_step}")
            
            if self.time_step >= self.horizon:
                self.is_success = -1
                break

            if current_target is None:
                grip_open = self.grip_open[self.current_stage]

                if self.current_stage == 0:
                    if use_gt:
                        trans_pre, rotation_pre = self.gt_actions[0]
                        trans_pre, rotation_pre = self._y_up_to_z_up(position=trans_pre, rotation=rotation_pre)
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
            
            if position_reached( self.c_controller, current_target[0], self.robot, thres=(0.1 if self.current_stage == 1 else 0.5) ) \
            and rotation_reached( self.c_controller, current_target[1] ):
                gripper_state = self.gripper_controller.get_joint_positions()
                current_gripper_open = (gripper_state[0] + gripper_state[1] > 7)
                
                if current_target[2] != current_gripper_open:
                    if current_target[2] < 0.5:
                        target_joint_positions_gripper = self.gripper_controller.forward(action="close")
                        for _ in range(self.cfg.gripper_trigger_period):
                            articulation_controller = self.robot.get_articulation_controller()
                            articulation_controller.apply_action(target_joint_positions_gripper)
                            self.try_record(actions=target_joint_positions_gripper)
                            simulation_context.step(render=render)

                    else:
                        target_joint_positions_gripper = self.gripper_controller.forward(action="open")
                        for _ in range(self.cfg.gripper_trigger_period):
                            articulation_controller = self.robot.get_articulation_controller()
                            articulation_controller.apply_action(target_joint_positions_gripper)
                            self.try_record(actions=target_joint_positions_gripper)
                            simulation_context.step(render=render)

                current_target = None
                self.current_stage += 1
                self.logger.info(f"enter stage {self.current_stage}")
            
            else:
                target_joint_positions = self.c_controller.forward(
                    target_end_effector_position=current_target[0], target_end_effector_orientation=current_target[1]
                )
                if self.current_stage >= 2:
                    # close force
                    target_joint_positions.joint_positions[-2:] = -1
                
                articulation_controller = self.robot.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions)
                self.try_record(actions=target_joint_positions)

            simulation_context.step(render=render)
            self.time_step += 1

        if self.current_stage == self.num_stages:
            # stages exhausted, success check
            for _ in range(self.cfg.success_check_period):
                simulation_context.step(render=False)
                if self.checker.success:
                    self.is_success = 1
                    break
        
        return self.render(), self.is_success
