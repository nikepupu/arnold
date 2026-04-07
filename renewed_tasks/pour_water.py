from .base_task import BaseTask
from typing import List
from environment.parameters import *
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path, get_all_matching_child_prims
from renewed_utils.semantics import add_update_semantics
from isaacsim.core.utils.types import ArticulationAction

import omni
import torch
from isaacsim.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from environment.fluid_utils import set_particle_system_for_cup
from local_utils.env import position_reached, rotation_reached, get_pre_grasp_action, action_interpolation
from local_utils.transforms import get_pose_relat, euler_angles_to_quat, quat_to_rot_matrix, matrix_to_quat, quat_diff_rad

import isaaclab.sim as sim_utils

from pxr import Gf, Sdf, UsdGeom, UsdShade
import logging
import numpy as np


class PourWater(BaseTask):
    def __init__(self, num_stages, horizon, stage_properties, record=False) -> None:
        super().__init__(num_stages, horizon, stage_properties, record)
        self.task = 'pour_water'
        self.gripper_trigger_period = 50
        self.success_check_period = 300
        self.grip_open = [True, False, False, False, False, False]
        self.logger = logging.getLogger(__name__)
        self.use_gpu_physics = True
        self.iso_surface = False

    def reset(self, robot_parameters,
              scene_parameters,
              object_parameters,
              robot_base,
              gt_actions,
        ):

        super().stop()

        self.robot_parameters: RobotParameters = robot_parameters
        self.object_parameter: ObjectParameters = object_parameters[0]
        self.stage = omni.usd.get_context().get_stage()
        self.checker = None

        self.robot_base = robot_base

        obs = super().reset(
            robot_parameters=robot_parameters,
            scene_parameters=scene_parameters
        )
        self.current_stage = 0
        self.end_stage = 0

        self.time_step = 0
        self.is_success = 0
        self.gt_actions = gt_actions

        return obs

    def set_up_task(self) -> None:
        self.load_object()

    def clear(self):
        super().clear()
        self._remove_particle_prims()

    def _remove_particle_prims(self):
        """Fully remove particle system prims from the USD layer.

        ClearReferences() leaves the prim alive, which causes
        particleUtils.add_physx_particle_system to fail its assertion
        on subsequent episodes.  Use the Sdf layer API to delete the
        prim spec so the prim no longer exists on the stage.
        """
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer()
        for path in ['/World_0/Fluid', '/World_0/Particles']:
            if not is_prim_path_valid(path):
                continue
            sdf_path = Sdf.Path(path)
            prim_spec = root_layer.GetPrimAtPath(sdf_path)
            if prim_spec:
                parent_spec = prim_spec.nameParent
                if parent_spec:
                    del parent_spec.nameChildren[prim_spec.name]

    def load_object(self):
        self.objects_list = []
        param = self.object_parameter

        object_prim_path, object_prim = self._prepare_object_prim(0, param.usd_path)

        cup_water_init_holder = object_prim_path
        cup_water_final_holder = object_prim_path

        particle_system_path = '/World_0/Fluid'
        particle_instance_str = "/World_0/Particles"

        volume_mesh_path = object_prim.GetPath().AppendPath("cup_volume").pathString
        self.objects_list.append(object_prim)

        positions = torch.tensor(np.array(param.object_position) / 100.0).unsqueeze(0)
        rotations = torch.tensor(param.orientation_quat).unsqueeze(0)
        scales = torch.tensor(np.array(param.scale) / 100.0).unsqueeze(0)

        XFormPrim(object_prim_path, positions=positions, orientations=rotations, scales=scales)
        self._wait_for_loading()

        mug_pos = np.array(param.object_position) / 100.0
        set_particle_system_for_cup(
            self.stage, Gf.Vec3f(mug_pos[0], mug_pos[1], mug_pos[2]),
            volume_mesh_path, particle_system_path, particle_instance_str,
            param.fluid_properties, asset_root="./asset",
            enable_iso_surface=self.iso_surface,
            distance_scale=0.01
        )

        self._wait_for_loading()

        self._make_cup_opaque(object_prim_path)

        if param.object_physics_properties:
            set_physics_properties(self.stage, object_prim, param.object_physics_properties)

        add_update_semantics(object_prim, param.object_type)

        if param.part_physics_properties:
            for keyword, properties in param.part_physics_properties.items():
                prim_list = get_all_matching_child_prims(object_prim_path, properties.properties[PREDICATE])
                for sub_prim_path in prim_list:
                    try:
                        sub_prim = get_prim_at_path(sub_prim_path)
                    except:
                        sub_prim = get_prim_at_path(sub_prim_path.GetPath().pathString)
                    set_physics_properties(self.stage, sub_prim, properties)
                    add_update_semantics(sub_prim, keyword)

        if param.object_timeline_management is not None:
            self.checker = param.object_timeline_management
            self.checker.pre_initialize(cup_water_init_holder, cup_water_final_holder,
                                        particle_instance_str, self.iso_surface)

    def _make_cup_opaque(self, object_prim_path):
        cup_prim = self.stage.GetPrimAtPath(object_prim_path)
        if not cup_prim:
            return
        for child in cup_prim.GetAllChildren():
            if child.GetTypeName() == "Mesh":
                gprim = UsdGeom.Gprim(child)
                gprim.CreateDisplayOpacityAttr([1.0])
            mat_bind = UsdShade.MaterialBindingAPI(child)
            mat = mat_bind.GetDirectBinding().GetMaterial()
            if mat:
                for shader in mat.GetPrim().GetAllChildren():
                    opacity_attr = shader.GetAttribute("inputs:opacity_constant")
                    if opacity_attr and opacity_attr.Get() is not None:
                        opacity_attr.Set(1.0)
                    enable_opacity = shader.GetAttribute("inputs:enable_opacity")
                    if enable_opacity and enable_opacity.Get() is not None:
                        enable_opacity.Set(False)

    def _apply_gripper_action(self, simulation_context, render, open_gripper):
        num_dofs = self.robot.num_dof
        gripper_indices = [num_dofs - 2, num_dofs - 1]
        if open_gripper:
            gripper_positions = np.array([0.05, 0.05])
        else:
            gripper_positions = np.array([0.0, 0.0])
        action = ArticulationAction(
            joint_positions=gripper_positions, joint_indices=gripper_indices
        )
        for _ in range(self.gripper_trigger_period):
            articulation_controller = self.robot.get_articulation_controller()
            articulation_controller.apply_action(action)
            self.try_record(actions=action)
            simulation_context.step(render=render)

    def step(self, act_pos, act_rot, render, use_gt):
        simulation_context = sim_utils.SimulationContext.instance()
        position_rotation_interp_list = None
        current_target = None

        if self.current_stage == 0:
            self._apply_gripper_action(simulation_context, render, open_gripper=True)

            self.end_stage = 2
            if use_gt:
                self.trans_pick, self.rotat_pick = self.gt_actions[1]
                self.trans_pick = np.array(self.trans_pick) / 100.0
            else:
                self.trans_pick = act_pos
                self.rotat_pick = act_rot
        else:
            self.end_stage = self.num_stages
            if use_gt:
                self.trans_target, self.rotat_target = self.gt_actions[2]
                self.trans_target = np.array(self.trans_target) / 100.0
            else:
                self.trans_target = act_pos
                self.rotat_target = act_rot

            up_rot_quat = euler_angles_to_quat(np.array([np.pi, 0, 0]))
            _, down_rot_mat = get_pose_relat(
                trans=None, rot=quat_to_rot_matrix(self.rotat_target),
                robot_pos=self.robot_base[0],
                robot_rot=quat_to_rot_matrix(self.robot_base[1])
            )
            down_rot_quat = matrix_to_quat(down_rot_mat)
            quat_diff = quat_diff_rad(up_rot_quat, down_rot_quat)
            num_interpolation = int(200 * quat_diff / (0.7 * np.pi))
            alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
            position_rotation_interp_list = action_interpolation(
                self.trans_pick, self.rotat_pick,
                self.trans_target, self.rotat_target,
                alphas, self.task
            )
            position_rotation_interp_iter = iter(position_rotation_interp_list)

        stage_step = 0
        stall = {"last_pos": None, "stall_count": 0}

        while self.current_stage < self.end_stage:
            if self.time_step % 120 == 0:
                self.logger.info(f"tick: {self.time_step}")

            if self.time_step >= self.horizon:
                self.is_success = -1
                break

            if self._check_stall(stall, stage_step):
                print(f'[pour_water] stage {self.current_stage} stalled, advancing',
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
                        trans_pre = np.array(trans_pre) / 100.0
                    else:
                        trans_pre, rotation_pre = get_pre_grasp_action(
                            grasp_action=(self.trans_pick, self.rotat_pick),
                            robot_base=self.robot_base, task=self.task
                        )
                    current_target = (trans_pre, rotation_pre, grip_open)

                elif self.current_stage == 1:
                    current_target = (self.trans_pick, self.rotat_pick, grip_open)

                elif self.current_stage == 2:
                    current_target = (
                        np.array([self.trans_pick[0], self.trans_target[1], self.trans_pick[2]]),
                        self.rotat_pick,
                        grip_open
                    )

                elif self.current_stage == 3:
                    current_target = (self.trans_target, self.rotat_pick, grip_open)

                elif self.current_stage == 4:
                    try:
                        trans_interp, rotation_interp = next(position_rotation_interp_iter)
                        current_target = (trans_interp, rotation_interp, grip_open)
                    except StopIteration:
                        position_rotation_interp_iter_back = iter(
                            position_rotation_interp_list[::-50] + position_rotation_interp_list[0:1]
                        )
                        self.current_stage += 1
                        continue

                elif self.current_stage == 5:
                    try:
                        trans_interp, rotation_interp = next(position_rotation_interp_iter_back)
                        current_target = (trans_interp, rotation_interp, grip_open)
                    except StopIteration:
                        position_rotation_interp_list = None
                        self.current_stage += 1
                        continue

            if (position_reached(self.c_controller, current_target[0], self.robot,
                                 thres=(0.002 if self.current_stage == 1 else 0.005))
                    or (self.current_stage in [4, 5])) \
                    and rotation_reached(self.c_controller, current_target[1]):

                joint_positions = self.robot.get_joint_positions()
                gripper_state = joint_positions[-2:]
                current_gripper_open = (gripper_state[0] + gripper_state[1] > 0.07)

                if current_target[2] != current_gripper_open:
                    self._apply_gripper_action(simulation_context, render,
                                              open_gripper=current_target[2])

                current_target = None
                if self.current_stage < 4:
                    self.current_stage += 1
                    stage_step = 0
                    stall = {"last_pos": None, "stall_count": 0}
                    self.logger.info(f"enter stage {self.current_stage}")

            else:
                target_joint_positions = self.c_controller.forward(
                    target_end_effector_position=current_target[0],
                    target_end_effector_orientation=current_target[1]
                )
                articulation_controller = self.robot.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions)
                self.try_record(actions=target_joint_positions)

            simulation_context.step(render=render)
            self.time_step += 1
            stage_step += 1

        for _ in range(self.success_check_period):
            simulation_context.step(render=False)
            if self.checker.success:
                self.is_success = 1
                break

        return self.render(), self.is_success
