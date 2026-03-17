from .pour_water import PourWater
from typing import List
from environment.parameters import *
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import is_prim_path_valid, get_prim_at_path, get_all_matching_child_prims
from isaacsim.core.utils.semantics import add_update_semantics

import omni
import torch
from isaacsim.core.prims import XFormPrim
from environment.physics_utils import set_physics_properties
from environment.fluid_utils import set_particle_system_for_cup
from pxr import Gf
import logging
import numpy as np

import isaaclab.sim as sim_utils


class TransferWater(PourWater):
    def __init__(self, num_stages, horizon, stage_properties, record=False) -> None:
        super().__init__(num_stages, horizon, stage_properties, record)
        self.task = 'transfer_water'
        self.grip_open = [True, False, False, False, False, False]
        self.logger = logging.getLogger(__name__)

    def reset(self, robot_parameters,
              scene_parameters,
              object_parameters,
              robot_base,
              gt_actions,
        ):
        self.objects_parameters: List[ObjectParameters] = object_parameters
        obs = super().reset(
            robot_parameters=robot_parameters,
            scene_parameters=scene_parameters,
            object_parameters=object_parameters,
            robot_base=robot_base,
            gt_actions=gt_actions
        )

        return obs

    def clear(self):
        super().clear()

    def load_object(self):
        index = 0
        self.objects_list = []
        cup_water_init_holder = None
        cup_water_final_holder = None
        particle_instance_str = "/World_0/Particles"
        particle_system_path = '/World_0/Fluid'

        import ipdb; ipdb.set_trace()

        for param in self.objects_parameters:
            object_prim_path = find_unique_string_name(
                initial_name=f"/World_{index}/{param.object_type}",
                is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            object_prim = add_reference_to_stage(param.usd_path, object_prim_path)
            volume_mesh_path = object_prim.GetPath().AppendPath("cup_volume").pathString

            self.objects_list.append(object_prim)

            positions = torch.tensor(np.array(param.object_position) / 100.0).unsqueeze(0)
            rotations = torch.tensor(param.orientation_quat).unsqueeze(0)
            scales = torch.tensor(np.array(param.scale) / 100.0).unsqueeze(0)

            XFormPrim(object_prim_path, positions=positions, orientations=rotations, scales=scales)
            self._wait_for_loading()

            if param.fluid_properties:
                cup_water_init_holder = object_prim_path

                mug_pos = np.array(param.object_position) / 100.0
                set_particle_system_for_cup(
                    self.stage, Gf.Vec3f(mug_pos[0], mug_pos[1], mug_pos[2]),
                    volume_mesh_path, particle_system_path, particle_instance_str,
                    param.fluid_properties, asset_root="./asset",
                    enable_iso_surface=self.iso_surface,
                    distance_scale=0.01
                )
            else:
                cup_water_final_holder = object_prim_path
                self.stage.GetPrimAtPath(volume_mesh_path).SetActive(False)

            self._wait_for_loading()

            self._make_cup_opaque(object_prim_path)

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

        for param in self.objects_parameters:
            if param.object_timeline_management is not None:
                self.checker = param.object_timeline_management
                self.checker.pre_initialize(cup_water_final_holder, cup_water_init_holder,
                                            particle_instance_str, self.iso_surface)
