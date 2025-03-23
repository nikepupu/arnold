from .base_task import BaseTask
from typing import List
from environment.parameters import *
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path, delete_prim, get_all_matching_child_prims
from omni.isaac.core.utils.semantics import add_update_semantics

import omni
from omni.isaac.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from utils.env import position_reached, rotation_reached, get_pre_grasp_action, action_interpolation
from omni.isaac.core.simulation_context import SimulationContext
from environment.fluid_utils import set_particle_system_for_cup
from utils.transforms import get_pose_relat, euler_angles_to_quat, quat_to_rot_matrix, matrix_to_quat, quat_diff_rad

from pxr import Gf
import logging
from omni.physx.scripts.utils import setStaticCollider
from omni.kit.material.library import get_material_prim_path
from pxr import UsdPhysics, Gf, PhysxSchema, UsdShade
from omni.isaac.core.utils.stage import set_stage_units, set_stage_up_axis


from typing import List, Optional

import carb
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.nucleus import get_assets_root_path


class Franka(Robot):
    """[summary]

    Args:
        prim_path (str): [description]
        name (str, optional): [description]. Defaults to "franka_robot".
        usd_path (Optional[str], optional): [description]. Defaults to None.
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
        end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
        gripper_dof_names (Optional[List[str]], optional): [description]. Defaults to None.
        gripper_open_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        gripper_closed_position (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "franka_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            if usd_path:
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            else:
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_rightfinger"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.04, 0.04]) 
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        else:
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/panda_rightfinger"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
            if gripper_dof_names is None:
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
            if gripper_open_position is None:
                gripper_open_position = np.array([0.04, 0.04]) 
            if gripper_closed_position is None:
                gripper_closed_position = np.array([0.0, 0.0])
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None, scale=np.array([100, 100, 100])
        )
        if gripper_dof_names is not None:
            if deltas is None:
                deltas = np.array([0.04, 0.04])
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position*100,
                joint_closed_positions=gripper_closed_position*100,
                action_deltas=deltas*100,
            )
        # change USD joint limits

        
        return

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> ParallelGripper:
        """[summary]

        Returns:
            ParallelGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]"""
        super().initialize(physics_sim_view)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._gripper.post_reset()
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[0], mode="position"
        )
        self._articulation_controller.switch_dof_control_mode(
            dof_index=self.gripper.joint_dof_indicies[1], mode="position"
        )
        return


class PourWater(BaseTask):
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        super().__init__(num_stages, horizon, stage_properties, cfg)
        self.task = 'pour_water'
        self.grip_open = cfg.gripper_open[self.task]
        self.logger = logging.getLogger(__name__)
        self.use_gpu_physics = True
        self.iso_surface = cfg.iso_surface

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
            robot_parameters = robot_parameters,
            scene_parameters = scene_parameters
        )
        # simulation_context = SimulationContext.instance() 
        # while True:
        #         simulation_context.step(render=True)
        # used for max mumber of steps (grasp, raise up)
        self.current_stage = 0
        self.end_stage = 0

        self.time_step = 0
        self.is_success = 0
        self.gt_actions = gt_actions

        return obs

    def set_up_task(self) -> None:
        self.load_object()

    def _load_scene(self):
        index = 0
        house_prim_path = f"/World_{index}/house"
        # print("house usd path: ", self.scene_parameters[index].usd_path)
        # while True:
        
        house_prim = add_reference_to_stage(self.scene_parameters[index].usd_path, house_prim_path)
        self._wait_for_loading()
        furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")
        
        # print(euler_angles_to_quat(np.array([np.pi/2, 0, 0])) )
        # house_prim.set_local_pose(np.array([0,0,0]) )
        # house_prim.set_local_pose(np.array([0,0,0]),  euler_angles_to_quat(np.array([np.pi/2, 0, 0])) )

        furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        #TODO 
        # somehow setting this is convexhull sometimes will lead to crash in omniverse
        setStaticCollider(furniture_prim, approximationShape=CONVEXHULL)
    
        self._wait_for_loading()

        room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")

        #TODO 
        # somehow setting this to convexHull will lead to some bug need to modify meshes later
        setStaticCollider(room_struct_prim, approximationShape="none")

        floor_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].floor_path}")
        self._set_ground_plane(index)
        
        wall_material_url = self.scene_parameters[index].wall_material_url
        floor_material_url = self.scene_parameters[index].floor_material_url
        
        if wall_material_url and floor_material_url:
            #TODO
            # this needs some refactor 
            wall_mtl_name = wall_material_url.split("/")[-1][:-4]
            floor_mtl_name = floor_material_url.split("/")[-1][:-4]
            if wall_mtl_name not in BaseTask.material_library:
                _, wall_material_prim_path = get_material_prim_path(wall_mtl_name)
                BaseTask.material_library[wall_mtl_name] = wall_material_prim_path
            else:
                wall_material_prim_path = BaseTask.material_library[wall_mtl_name]
            
            if floor_mtl_name not in BaseTask.material_library:
                _, floor_material_prim_path = get_material_prim_path(floor_mtl_name)
                BaseTask.material_library[floor_mtl_name] = floor_material_prim_path
            else:
                floor_material_prim_path = BaseTask.material_library[floor_mtl_name]
            
            # print("floor_material_url: ", floor_material_url)
            if floor_material_prim_path:
                # self._assets_root_path = get_assets_root_path()
                # print("load floor material")
                omni.kit.commands.execute(
                    "CreateMdlMaterialPrim",
                    mtl_url=floor_material_url,
                    mtl_name=floor_mtl_name,
                    mtl_path=floor_material_prim_path,
                    select_new_prim=False,
                )
                self._wait_for_loading()
                # print("created floor material")
                omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=floor_prim.GetPath(),
                    material_path=floor_material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants
                )
                self._wait_for_loading()
                # print("load floor material done")
            
            # print("wall_material_url: ", wall_material_url)
            if wall_material_prim_path:
                # print("load wall material")
                omni.kit.commands.execute(
                    "CreateMdlMaterialPrim",
                    mtl_url=wall_material_url,
                    mtl_name=wall_mtl_name,
                    mtl_path=wall_material_prim_path,
                    select_new_prim=False,
                )
                
                self._wait_for_loading()
                # print("created wall material")

                omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=room_struct_prim.GetPath(),
                    material_path=wall_material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants
                )
                
                self._wait_for_loading()
                # print("load wall material done")
        
        self._wait_for_loading()

    def _define_stage_properties(self):
        set_stage_up_axis(self.stage_properties.scene_up_axis)
        set_stage_units(0.01)
        self._set_up_physics_secne()
        
        skylight_path = '/skylight'
        add_reference_to_stage(self.stage_properties.light_usd_path, skylight_path)

    def _set_up_physics_secne(self):
        # reference : https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_transfer_policy.html
        physicsScenePath = "/physicsScene"
        scene = UsdPhysics.Scene.Get(self.stage, physicsScenePath)
        if not scene:
            scene = UsdPhysics.Scene.Define(self.stage, physicsScenePath)
        
        gravityDirection = self.stage_properties.gravity_direction
        self._gravityDirection = Gf.Vec3f(gravityDirection[0], gravityDirection[1],  gravityDirection[2])

        scene.CreateGravityDirectionAttr().Set(self._gravityDirection)

        self._gravityMagnitude = 9.81
        scene.CreateGravityMagnitudeAttr().Set(self._gravityMagnitude)
        
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)
        physxSceneAPI.GetTimeStepsPerSecondAttr().Set(120)
        physxSceneAPI.CreateEnableGPUDynamicsAttr().Set(self.use_gpu_physics )
        physxSceneAPI.CreateEnableEnhancedDeterminismAttr().Set(True)
        physxSceneAPI.CreateEnableStabilizationAttr().Set(True)

        physxSceneAPI.GetGpuMaxRigidContactCountAttr().Set(524288)
        physxSceneAPI.GetGpuMaxRigidPatchCountAttr().Set(81920)
        physxSceneAPI.GetGpuFoundLostPairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuFoundLostAggregatePairsCapacityAttr().Set(262144)
        physxSceneAPI.GetGpuTotalAggregatePairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuMaxSoftBodyContactsAttr().Set(1048576)
        physxSceneAPI.GetGpuMaxParticleContactsAttr().Set(1048576)
        # physxSceneAPI.GetGpuHeapCapacityAttr().Set(67108864)

    def _load_robot(self):
        # using one environment for now
        index = 0
        prim_path = f"/World_{index}/franka"

        position = self.robot_parameters[index].robot_position
        rotation = self.robot_parameters[index].robot_orientation_quat
        
        # position, rotation = self._y_up_to_z_up(position=position, rotation=rotation)

        robot = Franka(
                prim_path = prim_path, name = f"my_frankabot{index}",
                # usd_path = self.robot_parameters[index].usd_path,
                orientation = rotation,
                position = position,
                end_effector_prim_name = 'panda_rightfinger',
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
            )
        
        add_update_semantics(get_prim_at_path(prim_path), "Robot")
        prim_path =  f"/World_0/franka/panda_hand/panda_finger_joint1"
        joint = UsdPhysics.PrismaticJoint.Get(self.stage, prim_path)	
        if joint:
            upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
            joint.CreateUpperLimitAttr(upper_limit * 100  )
        
        prim_path =  f"/World_0/franka/panda_hand/panda_finger_joint2"
        joint = UsdPhysics.PrismaticJoint.Get(self.stage, prim_path)	
        if joint:
            upper_limit = joint.GetUpperLimitAttr().Get() #GetAttribute("xformOp:translate").Get()
            joint.CreateUpperLimitAttr(upper_limit * 100 )

        self._wait_for_loading()
        self._set_sensors()
     
        return robot

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

        object_prim = add_reference_to_stage(param.usd_path, object_prim_path)

        cup_water_init_holder = object_prim_path
        cup_water_final_holder = object_prim_path

        particle_system_path = '/World_0/Fluid'

        particle_instance_str = "/World_0/Particles"
        
        volume_mesh_path = object_prim.GetPath().AppendPath(f"cup_volume").pathString
        self.objects_list.append(object_prim)
        
        position = np.array(param.object_position)
        rotation = param.orientation_quat
        
        # use this to set relative position, orientation and scale
        xform_prim = XFormPrim(object_prim_path, translation= position, orientation = rotation, scale = np.array(param.scale))
      
        self._wait_for_loading()

        set_particle_system_for_cup(
            self.stage, Gf.Vec3f(position[0], position[1], position[2]),
            volume_mesh_path, particle_system_path, particle_instance_str, param.fluid_properties,
            asset_root=self.cfg.asset_root, enable_iso_surface=self.iso_surface
        )

        self._wait_for_loading()

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
                        # since 2022.1.1 get_prim_at_path returns a prim instead of a path
                        sub_prim = get_prim_at_path(sub_prim_path.GetPath().pathString)
                    set_physics_properties(self.stage, sub_prim, properties)
                    add_update_semantics(sub_prim, keyword)
        
        if param.object_timeline_management is not None:
            self.checker = param.object_timeline_management
            self.checker.pre_initialize(cup_water_init_holder, cup_water_final_holder, particle_instance_str, self.iso_surface)
    
    def remove_objects(self):
        for prim in self.objects_list:
            delete_prim(prim.GetPath().pathString)
        if is_prim_path_valid('/World_0/Fluid'):
            delete_prim('/World_0/Fluid')
        if is_prim_path_valid('/World_0/Particles'):
            delete_prim("/World_0/Particles")
        if is_prim_path_valid('/Looks/Water'):
            delete_prim("/Looks/Water")
        if is_prim_path_valid('/World'):
            delete_prim('/World')
        if is_prim_path_valid('/Looks'):
            delete_prim('/Looks')
        if is_prim_path_valid('/lula'):
            delete_prim('/lula')
        self.objects_list = []
        self._wait_for_loading()

    def step(self, act_pos, act_rot, render, use_gt):
        """
        `act_pos`: np.ndarray (3,)
        `act_rot`: np.ndarray (4,) (wxyz)
        `render`: bool
        `use_gt`: bool
        `step` is called twice, first for grasping object and second for manipulating object
        """
        simulation_context = SimulationContext.instance()
        position_rotation_interp_list = None
        current_target = None

        if self.current_stage == 0:
            target_joint_positions_gripper = self.gripper_controller.forward(action="open")
            for _ in range(self.cfg.gripper_trigger_period):
                articulation_controller = self.robot.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions_gripper)
                self.try_record(actions=target_joint_positions_gripper)
                simulation_context.step(render=render)


            self.end_stage = 2
            if use_gt:
                self.trans_pick, self.rotat_pick = self.gt_actions[1]
                self.trans_pick = np.array(self.trans_pick)/1.0
            else:
                self.trans_pick = act_pos
                self.rotat_pick = act_rot
        else:
            self.end_stage = self.num_stages
            if use_gt:
                self.trans_target, self.rotat_target = self.gt_actions[2]
                self.trans_target = np.array(self.trans_target)/1.0
            else:
                self.trans_target = act_pos
                self.rotat_target = act_rot

            # interpolation for manipulation
            up_rot_quat = euler_angles_to_quat(np.array([np.pi, 0, 0]))
            _, down_rot_mat = get_pose_relat(
                trans=None, rot=quat_to_rot_matrix(self.rotat_target),
                robot_pos=self.robot_base[0],
                robot_rot=quat_to_rot_matrix(self.robot_base[1])
            )
            down_rot_quat = matrix_to_quat(down_rot_mat)
            quat_diff = quat_diff_rad(up_rot_quat, down_rot_quat)
            num_interpolation = int(200 * quat_diff / (0.7*np.pi))
            alphas = np.linspace(start=0, stop=1, num=num_interpolation)[1:]
            position_rotation_interp_list = action_interpolation(
                self.trans_pick, self.rotat_pick, self.trans_target, self.rotat_target, alphas, self.task
            )
            position_rotation_interp_iter = iter(position_rotation_interp_list)

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
                        trans_pre = np.array(trans_pre)/1.0
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
                    except:
                        # finish interpolation
                        position_rotation_interp_iter_back = iter(
                            position_rotation_interp_list[::-50] + position_rotation_interp_list[0:1]
                        )
                        self.current_stage += 1
                        continue

                elif self.current_stage == 5:
                    # cup return to upward orientation
                    try:
                        trans_interp, rotation_interp = next(position_rotation_interp_iter_back)
                        current_target = (trans_interp, rotation_interp, grip_open)
                    except:
                        # finish interpolation
                        position_rotation_interp_list = None
                        self.current_stage += 1
                        continue

            if ( position_reached(self.c_controller, current_target[0], self.robot, thres=(0.1 if self.current_stage == 1 else 0.5)) or (self.current_stage in [4,5]) ) \
            and rotation_reached(self.c_controller, current_target[1]):
                gripper_state = self.gripper_controller.get_joint_positions()
                current_gripper_open = (gripper_state[0] + gripper_state[1] > 7.0)

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
                if self.current_stage < 4:
                    self.current_stage += 1
                    self.logger.info(f"enter stage {self.current_stage}")
            
            else:
                target_joint_positions = self.c_controller.forward(
                    target_end_effector_position=current_target[0], target_end_effector_orientation=current_target[1]
                )
                
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
