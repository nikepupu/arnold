
from environment.parameters import *
from local_utils.recorder import DataRecorder

import omni
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot.manipulators.examples")
enable_extension("isaacsim.sensors.camera")

import isaaclab.sim as sim_utils
import torch
import os

from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path, delete_prim
from isaacsim.robot.manipulators.examples.franka import Franka
from renewed_utils.semantics import add_update_semantics

from isaacsim.core.utils.stage import set_stage_units, set_stage_up_axis, is_stage_loading
from isaacsim.core.utils.stage import add_reference_to_stage
# from isaacsim.sensor import Camera
from omni.physx.scripts.utils import setStaticCollider
from omni.kit.material.library import get_material_prim_path

from omni.physx.scripts import physicsUtils
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.sensors.camera import Camera

import pxr
from pxr import UsdPhysics, Gf, PhysxSchema, UsdShade
from abc import ABC
# from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from renewed_utils.controller import RMPFlowController

from typing import List, Optional

import carb
import numpy as np
from isaacsim.core.prims import RigidPrim
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units, get_current_stage_id
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.core.simulation_manager import SimulationManager
import omni.physics.tensors

# from isaacsim.nucleus import get_assets_root_path


class BaseTask(ABC):
    material_library = {}
    viewport_handles = []
    
    def __init__(self, num_stages, horizon, stage_properties, record=False) -> None:
        self.record = record
        self.num_stages = num_stages
        self.horizon = horizon
        self.stage_properties: StageProperties = stage_properties
        self.simulation_context = sim_utils.SimulationContext.instance()
        # self.timeline = omni.timeline.get_timeline_interface()
        self.kit = omni.kit.app.get_app()

        self.objects_list = []
        self.recorder = None
        self._robot_loaded = False
        self._sensor_initialized = False

        self.gripper_trigger_period = 50
        self.success_check_period = 300
    
    def success(self):
        if hasattr(self, "checker") and self.checker and self.checker.success:
            return True
        
        return False

    def set_up_task(self):
        raise NotImplementedError

    def _get_ee_pos(self):
        """Return the current end-effector position as a numpy array."""
        pos, _ = self.c_controller.get_motion_policy().get_end_effector_as_prim().get_world_pose()
        return np.asarray(pos, dtype=np.float64)

    def _check_stall(self, stall_state, stage_step):
        """Detect if the robot end-effector is stalled.

        Args:
            stall_state: dict with keys 'last_pos' and 'stall_count',
                         mutated in place.
            stage_step:  steps elapsed in the current stage.

        Returns:
            True if the robot has been stalled long enough to skip.
        """
        check_interval = 60          # compare every 0.5 s at 120 Hz
        stall_threshold = 0.01       # 1 cm
        max_stall_checks = 10         # 5 consecutive stalls → give up

        if stage_step > 0 and stage_step % check_interval == 0:
            ee_pos = self._get_ee_pos()
            if stall_state["last_pos"] is not None:
                delta = np.linalg.norm(ee_pos - stall_state["last_pos"])
                if delta < stall_threshold:
                    stall_state["stall_count"] += 1
                else:
                    stall_state["stall_count"] = 0
            stall_state["last_pos"] = ee_pos

        return stall_state["stall_count"] >= max_stall_checks

    def remove_objects(self):
        self.objects_list = []

    def _kill_checker(self):
        """Deactivate the checker and force-unsubscribe its physics/timeline
        callbacks so they don't interfere with scene reconfiguration."""
        
        # self.simulation_context.pause()
        if hasattr(self, "checker") and self.checker:
            self.checker.is_init = False
            self.checker.reset()
            self.checker = None
        import gc
        gc.collect()
        # self.simulation_context.play()

    def _recreate_simulation_view(self):
        """Invalidate the stale global SimulationView cached in
        SimulationManager and create a fresh one that reflects the
        current USD stage composition."""
        if SimulationManager._physics_sim_view is not None:
            SimulationManager._physics_sim_view.invalidate()
            SimulationManager._physics_sim_view = None
        if SimulationManager._physics_sim_view__warp is not None:
            SimulationManager._physics_sim_view__warp.invalidate()
            SimulationManager._physics_sim_view__warp = None

        backend = SimulationManager.get_backend()
        stage_id = get_current_stage_id()

        SimulationManager._physics_sim_view = omni.physics.tensors.create_simulation_view(
            backend, stage_id=stage_id
        )
        SimulationManager._physics_sim_view.set_subspace_roots("/")
        SimulationManager._physics_sim_view__warp = omni.physics.tensors.create_simulation_view(
            "warp", stage_id=stage_id
        )
        SimulationManager._physics_sim_view__warp.set_subspace_roots("/")

    def stop(self):
        if self.recorder is not None and self.recorder.record:
            self.recorder.save_buffer(self.success())
            self.recorder = None

        self._kill_checker()
        self.clear()
        

    def reset(self,
              robot_parameters = None,
              scene_parameters = None,
              sensor_resolution = (128, 128),
              sensor_types = ["rgb", "depthLinear", "camera", "semanticSegmentation"],
        ):
        self._kill_checker()
        self.kit.update()

        self.stage = sim_utils.get_current_stage()
        self.sensor_resolution = sensor_resolution
        self.sensor_types = sensor_types

        if robot_parameters is not None:
            self.robot_parameters = robot_parameters

        if scene_parameters is not None:
            self.scene_parameters = scene_parameters
            self.num_envs = len(scene_parameters)

        self.clear()
        self._wait_for_loading()

        if not self._robot_loaded:
            self._define_stage_properties()
            self._load_scene()
            self.robot = self._load_robot()
            self._robot_loaded = True

            self.set_up_task()
            self._wait_for_loading()

            if not self.simulation_context.is_playing():
                self.simulation_context.play()
        else:
            self.simulation_context.pause()
            self.kit.update()

            self._load_scene()
            self._reposition_robot()

            self.set_up_task()
            self._wait_for_loading()

            omni.physx.get_physx_interface().force_load_physics_from_usd()
            self.kit.update()

            self.simulation_context.play()
            self._recreate_simulation_view()

        def initialize(robot):
            robot._articulation_view._physics_view = None
            robot._articulation_view._is_initialized = False
            robot.initialize()
            if not hasattr(self, '_default_joint_positions'):
                self._default_joint_positions = robot._articulation_view._default_joints_state.positions.clone()
                self._default_joint_velocities = robot._articulation_view._default_joints_state.velocities.clone()
                self._default_joint_efforts = robot._articulation_view._default_joints_state.efforts.clone()
            robot.set_joint_positions(self._default_joint_positions)
            robot.set_joint_velocities(self._default_joint_velocities)
            robot.set_joint_efforts(self._default_joint_efforts)
            _dp = np.asarray(self._default_joint_positions.cpu()).flatten().copy()
            articulation_controller = robot.get_articulation_controller()
            articulation_controller.apply_action(
                ArticulationAction(joint_positions=_dp)
            )
            add_update_semantics(get_prim_at_path(robot.prim_path), "Robot")
            robot.disable_gravity()
            self.kit.update()
            robot.set_joint_positions(self._default_joint_positions)
            robot.set_joint_velocities(self._default_joint_velocities)
            articulation_controller.apply_action(
                ArticulationAction(joint_positions=_dp)
            )

        initialize(self.robot)

        default_pos = self._default_joint_positions.clone()
        default_pos[..., -2:] = 0.05
        default_pos_np = np.asarray(default_pos.cpu()).flatten()
        articulation_controller = self.robot.get_articulation_controller()

        def _hold_default_pose(n_steps, render=False):
            for _ in range(n_steps):
                self.robot.set_joint_positions(default_pos)
                self.robot.set_joint_velocities(self._default_joint_velocities)
                articulation_controller.apply_action(
                    ArticulationAction(joint_positions=default_pos_np)
                )
                self.simulation_context.step(render=render)

        _hold_default_pose(self.gripper_trigger_period)

        if self.simulation_context is not None:
            for _ in range(240):
                self.simulation_context.step(render=False)

        if self.simulation_context is not None:
            _hold_default_pose(10)

        self.robot.set_joint_positions(default_pos)
        self.robot.set_joint_velocities(self._default_joint_velocities)
        articulation_controller.apply_action(
            ArticulationAction(joint_positions=default_pos_np)
        )

        self.time_step = 0
        self.gripper_controller = self.robot.gripper
        if hasattr(self, 'c_controller') and self.c_controller is not None:
            self.c_controller.reset()
            self.c_controller = None
        self.c_controller = RMPFlowController(name="cspace_controller", robot_articulation=self.robot, physics_dt=1/120.0)

        if self.record:
            self.register_recorder()

        for _ in range(100):
            self.simulation_context.render()

        if self.checker is not None:
            self.checker.initialization_step()

            _hold_default_pose(240, render=False)

            if hasattr(self.checker, 'read_settled_init_y'):
                self.checker.read_settled_init_y()

            self.robot._articulation_view._physics_view = None
            self.robot._articulation_view._is_initialized = False
            self.robot.initialize()
            self.robot.set_joint_positions(default_pos)
            self.robot.set_joint_velocities(self._default_joint_velocities)
            self.robot.set_joint_efforts(self._default_joint_efforts)
            articulation_controller = self.robot.get_articulation_controller()
            articulation_controller.apply_action(
                ArticulationAction(joint_positions=default_pos_np)
            )
            self.robot.disable_gravity()
            self.kit.update()
            self.robot.set_joint_positions(default_pos)
            self.robot.set_joint_velocities(self._default_joint_velocities)
            articulation_controller.apply_action(
                ArticulationAction(joint_positions=default_pos_np)
            )

        return self.render()

    def step(self):
        raise NotImplementedError

    def _define_stage_properties(self):
        set_stage_up_axis(self.stage_properties.scene_up_axis)
        set_stage_units(1.0)
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
        self._gravityDirection = Gf.Vec3f(gravityDirection[0], gravityDirection[1], gravityDirection[2])
        scene.CreateGravityDirectionAttr().Set(self._gravityDirection)

        self._gravityMagnitude = 9.81
        scene.CreateGravityMagnitudeAttr().Set(self._gravityMagnitude)

        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)
        physxSceneAPI.GetTimeStepsPerSecondAttr().Set(120)
        physxSceneAPI.CreateEnableGPUDynamicsAttr().Set(self.use_gpu_physics)
        if self.use_gpu_physics:
            physxSceneAPI.CreateBroadphaseTypeAttr().Set("GPU")
        physxSceneAPI.CreateEnableEnhancedDeterminismAttr().Set(True)
        physxSceneAPI.CreateEnableStabilizationAttr().Set(True)

        physxSceneAPI.GetGpuMaxRigidContactCountAttr().Set(524288)
        physxSceneAPI.GetGpuMaxRigidPatchCountAttr().Set(81920)
        physxSceneAPI.GetGpuFoundLostPairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuFoundLostAggregatePairsCapacityAttr().Set(262144)
        physxSceneAPI.GetGpuTotalAggregatePairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuMaxSoftBodyContactsAttr().Set(1048576)
        physxSceneAPI.GetGpuMaxParticleContactsAttr().Set(1048576)
        
    def render(self):
        if not self._sensor_initialized:
            return None

        self.simulation_context.render()
        gts = list(map(Camera.get_current_frame, self.cameras))
        outputs = []
        for i, gt in enumerate(gts):
            output = {'camera': self.camera_configs[i]}
            output['camera']['pose'] = self.cameras[i]._backend_utils.inverse(self.cameras[i].get_view_matrix_ros())
            if 'rgba' in gt:
                output['rgb'] = gt['rgba'].copy()
            if 'distance_to_image_plane' in gt:
                output['depthLinear'] = gt['distance_to_image_plane'].copy() if gt['distance_to_image_plane'] is not None else None
            if 'semantic_segmentation' in gt:
                output['semanticSegmentation'] = gt['semantic_segmentation'].copy() if gt['semantic_segmentation'] is not None else None
            outputs.append(output)

        return {'images': outputs}

    def clear(self):
        for prim in self.objects_list:
            if prim.IsValid():
                prim.GetReferences().ClearReferences()
        self.objects_list = []
    
    def _rescale_prismatic_joint_limits(self, object_prim_path, scale):
        """Scale prismatic joint limits to match the object's XFormPrim scale.

        USD joint limits are authored in the object's original coordinate space.
        When the object is scaled on stage, prismatic (linear) limits must be
        multiplied by the corresponding scale factor so the physical travel
        distance matches the scaled geometry.
        """
        from pxr import Usd, UsdPhysics
        scale_value = float(scale[0]) if hasattr(scale, '__len__') else float(scale)
        if abs(scale_value - 1.0) < 1e-6:
            return
        root = get_prim_at_path(object_prim_path)
        if not root or not root.IsValid():
            return
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.PrismaticJoint):
                joint = UsdPhysics.PrismaticJoint(prim)
                upper = joint.GetUpperLimitAttr().Get()
                lower = joint.GetLowerLimitAttr().Get()
                if upper is not None:
                    joint.GetUpperLimitAttr().Set(upper * scale_value)
                if lower is not None:
                    joint.GetLowerLimitAttr().Set(lower * scale_value)

    def _prepare_object_prim(self, slot, usd_path):
        """Delete the old task-object prim entirely and load the USD into a
        fresh prim so no stale physics state carries over."""
        prim_path = f"/World_0/task_object_{slot}"
        if is_prim_path_valid(prim_path):
            delete_prim(prim_path)
            self.kit.update()
        prim = add_reference_to_stage(usd_path, prim_path)
        return prim_path, prim

    def _load_scene(self):
        index = 0
        house_prim_path = f"/World_{index}/house"
        self.scene_parameters[index].usd_path = self.scene_parameters[index].usd_path.replace("/VRKitchen2.0", "")

        if is_prim_path_valid(house_prim_path):
            get_prim_at_path(house_prim_path).GetReferences().ClearReferences()

        house_prim = add_reference_to_stage(self.scene_parameters[index].usd_path, house_prim_path)
        self._wait_for_loading()
        furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")
          
        XFormPrim(house_prim_path, scales=[[0.01, 0.01, 0.01]])

        # furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        # setStaticCollider(furniture_prim, approximationShape=CONVEXHULL)
    
        self._wait_for_loading()

        # room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")
        # setStaticCollider(room_struct_prim, approximationShape="none")

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
                if not is_prim_path_valid(floor_material_prim_path):
                    omni.kit.commands.execute(
                        "CreateMdlMaterialPrim",
                        mtl_url=floor_material_url,
                        mtl_name=floor_mtl_name,
                        mtl_path=floor_material_prim_path,
                        select_new_prim=False,
                    )
                    self._wait_for_loading()
                omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=floor_prim.GetPath(),
                    material_path=floor_material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants
                )
                self._wait_for_loading()
            
            if wall_material_prim_path:
                if not is_prim_path_valid(wall_material_prim_path):
                    omni.kit.commands.execute(
                        "CreateMdlMaterialPrim",
                        mtl_url=wall_material_url,
                        mtl_name=wall_mtl_name,
                        mtl_path=wall_material_prim_path,
                        select_new_prim=False,
                    )
                    self._wait_for_loading()

                omni.kit.commands.execute(
                    "BindMaterial",
                    prim_path=room_struct_prim.GetPath(),
                    material_path=wall_material_prim_path,
                    strength=UsdShade.Tokens.strongerThanDescendants
                )
                
                self._wait_for_loading()
        
        self._wait_for_loading()

    def _set_ground_plane(self, index):
        ground_plane_path = f"/World_{index}/house/groundPlane"
        if not is_prim_path_valid(ground_plane_path):
            physicsUtils.add_ground_plane(self.stage, ground_plane_path, "Y", 5000.0,
                pxr.Gf.Vec3f(0.0, 0.0, 0.0), pxr.Gf.Vec3f(0.2))
        ground_prim = self.stage.GetPrimAtPath(ground_plane_path)
        ground_prim.GetAttribute('visibility').Set('invisible')

    def _load_robot(self):
        # using one environment for now
        index = 0
        prim_path = f"/World_{index}/franka"

        position = self.robot_parameters[index].robot_position
        rotation = self.robot_parameters[index].robot_orientation_quat
        
        # position, rotation = self._y_up_to_z_up(position=position, rotation=rotation)

        robot = Franka(
                prim_path = prim_path, name = f"my_frankabot{index}",
                # custom franka.usd is in cm scale; use default (meter-scale) Franka instead
                # usd_path = self.robot_parameters[index].usd_path,
                orientation = rotation,
                position = position / 100.0,
                end_effector_prim_name = 'panda_rightfinger',
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
            )

        robot_prim = get_prim_at_path(prim_path)
        physx_art_api = PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
        physx_art_api.CreateEnabledSelfCollisionsAttr().Set(False)

        from pxr import Usd
        _col_prims = []
        for _cp in Usd.PrimRange(robot_prim):
            if _cp.HasAPI(UsdPhysics.CollisionAPI):
                _col_prims.append(_cp.GetPath())
        if len(_col_prims) > 1:
            _fp_api = UsdPhysics.FilteredPairsAPI.Apply(robot_prim)
            _fp_api.GetFilteredPairsRel().SetTargets(_col_prims)

        # #region agent log
        import math, json as _json, time as _time
        _ARM_JOINTS = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                       "panda_joint5", "panda_joint6", "panda_joint7"}
        _EXT_DEG = math.degrees(0.2)
        _jlim_info = {}
        for _cp in Usd.PrimRange(robot_prim):
            if _cp.GetName() in _ARM_JOINTS:
                _lo_attr = _cp.GetAttribute("physics:lowerLimit")
                _hi_attr = _cp.GetAttribute("physics:upperLimit")
                if _lo_attr.IsValid() and _hi_attr.IsValid():
                    _old_lo, _old_hi = _lo_attr.Get(), _hi_attr.Get()
                    _lo_attr.Set(_old_lo - _EXT_DEG)
                    _hi_attr.Set(_old_hi + _EXT_DEG)
                    _jlim_info[_cp.GetName()] = {"old": [round(_old_lo,2), round(_old_hi,2)],
                                                  "new": [round(_old_lo - _EXT_DEG,2), round(_old_hi + _EXT_DEG,2)]}
        with open('/home/rgong/Desktop/arnold/.cursor/debug-5787ac.log', 'a') as _f:
            _f.write(_json.dumps({"sessionId":"5787ac","location":"base_task.py:_load_robot","message":"joint_limits_extended",
                                  "data":_jlim_info,"timestamp":int(_time.time()*1000),"hypothesisId":"AJ"}) + '\n')
        # #endregion

        add_update_semantics(robot_prim, "Robot")
        self._wait_for_loading()
        # self._set_sensors()
     
        return robot

    def _reposition_robot(self):
        index = 0
        position = self.robot_parameters[index].robot_position
        rotation = self.robot_parameters[index].robot_orientation_quat
        # Use USD-level XFormPrim instead of physics-level set_world_pose,
        # because the articulation physics view may be stale after scene
        # changes (ClearReferences) while the simulation is still running.
        # Joint states are reset later by initialize() in reset().
        XFormPrim(
            self.robot.prim_path,
            positions=torch.tensor(np.array(position, dtype=np.float64) / 100.0).unsqueeze(0),
            orientations=torch.tensor(np.array(rotation, dtype=np.float64)).unsqueeze(0),
        )

        self.c_controller._motion_policy.set_robot_base_pose(
            robot_position=np.array(position, dtype=np.float64) / 100.0,
            robot_orientation=np.array(rotation, dtype=np.float64)
        )


    # Camera transforms from the custom franka.usd (positions in cm, converted to meters).
    # Orientations are raw USD xform quaternions (USD camera convention: -Z forward, +Y up).
    CAMERA_SPECS = {
        'FrontCamera': {
            'translation': np.array([10, -0.000010252, 120]) / 100.0,
            'orientation': np.array([0.6830127, 0.18301270, -0.18301270, -0.6830127]),
        },
        'BaseCamera': {
            'translation': np.array([7.0169, 0, 25.0]) / 100.0,
            'orientation': np.array([0.5, 0.5, -0.5, -0.5]),
        },
        'LeftCamera': {
            'translation': np.array([20.0, 90, 80.0]) / 100.0,
            'orientation': np.array([0.0, 0.0, 0.5373, 0.8434]),
        },
        'GripperCameraBottom': {
            'translation': np.array([-6, 0, -3]) / 100.0,
            'orientation': np.array([0.0, -0.7071068, -0.7071068, 0.0]),
        },
        'GripperCamera': {
            'translation': np.array([4.9648, -0.0545, -1.9908]) / 100.0,
            'orientation': np.array([0.0723, 0.6594, 0.7461, 0.0564]),
        },
    }

    def _set_sensors(self):
        self._register_camera_path()
        BaseTask.cameras = []

        for idx, camera_path in enumerate(self.camera_paths):
            cam_name = camera_path.split('/')[-1]
            spec = self.CAMERA_SPECS.get(cam_name)

            camera = Camera(
                prim_path=camera_path, frequency=20,
                resolution=self.sensor_resolution,
            )
            if spec:
                camera.set_local_pose(
                    translation=spec['translation'],
                    orientation=spec['orientation'],
                    camera_axes="usd",
                )
            camera.set_focal_length(12.49)
            camera.set_clipping_range(0.001, 10000.0)
            camera.set_horizontal_aperture(20.955)
            camera.set_vertical_aperture(20.955)
            camera.initialize()
            for sensor_type in self.sensor_types:
                if 'depth' in sensor_type:
                    camera.add_distance_to_image_plane_to_frame()
                elif 'semantic' in sensor_type:
                    camera.add_semantic_segmentation_to_frame()
            BaseTask.cameras.append(camera)

        self.kit.update()
        self._sensor_initialized = True

        self.camera_configs = []
        for camera in self.cameras:
            width, height = camera.get_resolution()
            camera_config = {
                'resolution': {'width': width, 'height': height},
                'focal_length': camera.get_focal_length(),
                'horizontal_aperture': camera.get_horizontal_aperture(),
            }
            self.camera_configs.append(camera_config)

    def _register_camera_path(self):
        self.camera_paths = []
        
        robot_path = f'/World_{0}/franka'
        camera_paths = [ 
            f'{robot_path}/FrontCamera', 
            f'{robot_path}/BaseCamera',
            f'{robot_path}/LeftCamera',
            f'{robot_path}/panda_hand/GripperCameraBottom',
            f'{robot_path}/panda_hand/GripperCamera'
        ]

        for camera_path in camera_paths:
            self.camera_paths.append(camera_path)

    def _wait_for_loading(self):
        if self.simulation_context.is_playing():
            self.simulation_context.step(render=True)
        else:
            self.kit.update()

    def register_recorder(self):
        index = 0
        objects_paths = [prim.GetPath().pathString for prim in self.objects_list]
        self.recorder = DataRecorder(self.robot.prim_path, objects_paths, self.robot, self.scene_parameters[index].task_type)

    def try_record(self, actions):
        if self.recorder is not None and self.recorder.record:
            # dof_states = self.dc.get_articulation_dof_states(self.articulation, _dynamic_control.STATE_ALL)
            dof_states = {
                'pos': self.robot.get_joint_positions(),
                'vel': self.robot.get_joint_velocities(),
                # 'effort': self.robot.get_joint_efforts(),   # get_measured_joint_efforts() for Isaac Sim 2023
                # getting effort has conflict with use_gpu_dynamics
            }
            self.recorder.record_data(
                robot_states=dof_states,
                actions=actions,
                time_step=self.time_step,
            )
