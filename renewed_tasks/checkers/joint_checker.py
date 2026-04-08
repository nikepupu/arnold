import math
import numpy as np
import omni
import torch
from isaacsim.core.prims import XFormPrim
# from isaacsim.dynamic_control import _dynamic_control
from pxr import UsdPhysics, PhysxSchema
from .base_checker import BaseChecker
from environment.parameters import CheckerParameters
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
from isaacsim.core.prims import Articulation

class JointCheck():
    def __init__(self, joint_prim, joint_name) -> None:
        self.joint_name = joint_name
        self.stage = sim_utils.get_current_stage()

        self.prim_list = list(self.stage.TraverseAll())
        # print("self.prim_list: ", self.prim_list)
        self.prim_list = [ item for item in self.prim_list if joint_name in  
            item.GetPath().pathString and item.GetPath().pathString.startswith(joint_prim) and item.GetPath().pathString.endswith(joint_name)]

        assert len(self.prim_list) == 1, "len of " + str(len(self.prim_list))
        self.prim = self.prim_list[0]

        self.type = self.prim.GetTypeName()
        self.full_name = self.prim.GetPath().pathString
        self.joint = self.stage.GetPrimAtPath(self.full_name)
        self.link = self.joint.GetRelationship("physics:body1").GetTargets()[0]


        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleGeometryPrim

        link_material = PhysicsMaterial(
            prim_path=f"/World_0/physics_material/link_material",
            name="link_material",
            static_friction=200,
            dynamic_friction=150,
            restitution=0.0,
        )

        link_geom = SingleGeometryPrim(
            prim_path=f"{self.link}",
            name=f"{self.link}_geom",
        )
        link_geom.apply_physics_material(link_material)

        # Determine the appropriate drive type name
        self.drive_type = None
        if self.prim.IsA(UsdPhysics.RevoluteJoint):
            self.drive_type = "angular"
        elif self.prim.IsA(UsdPhysics.PrismaticJoint):
            self.drive_type = "linear"

        drive_api = UsdPhysics.DriveAPI.Get(self.prim, self.drive_type)
        stiffness = drive_api.GetStiffnessAttr().Get()
        damping   = drive_api.GetDampingAttr().Get()
        print("stiffness: ", stiffness, "damping: ", damping)

        drive_api.GetStiffnessAttr().Set(0.0)
        drive_api.GetDampingAttr().Set(0.0)    

        # Get joint state api
        if not self.prim.HasAPI(PhysxSchema.JointStateAPI, self.drive_type):
            self.joint_state_api = PhysxSchema.JointStateAPI.Apply(self.prim, self.drive_type)
        else:
            self.joint_state_api = PhysxSchema.JointStateAPI.Get(self.prim, self.drive_type)

        # import ipdb; ipdb.set_trace()
        # self.parent = self.joint.GetRelationship("physics:body0").GetTargets()[0]
        # root_api = UsdPhysics.ArticulationRootAPI.Apply(
        #     self.stage.GetPrimAtPath(self.parent.pathString)
        # )
        self.articulation = Articulation(prim_paths_expr=joint_prim)


    def get_joint_position(self):
        body1 = str(self.joint.GetRelationship("physics:body1").GetTargets()[0])

        pos, rot = XFormPrim(body1).get_world_poses()
        
        # FIXME: correct numpy as tensor, if is a tensor, turn to numpy
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
        return pos[0] # only the first element
    
    def get_joint_link(self):
        body0 = self.joint.GetRelationship("physics:body0").GetTargets()[0]
        body1 = self.joint.GetRelationship("physics:body1").GetTargets()[0]
        return body1

    @property
    def upper(self):
        return self.joint.GetAttribute("physics:upperLimit").Get()
    
    @property
    def lower(self):
        return self.joint.GetAttribute("physics:lowerLimit").Get()
        
    def compute_percentage(self):
        #get the joint percentage and check
        # joint_postion = self.joint_state_api.GetPositionAttr().Get()
        joint_index = self.articulation.dof_names.index(self.joint_name)
        joint_position = self.articulation.get_joint_positions(joint_indices=[joint_index])
        
        if isinstance(joint_position, torch.Tensor):
            joint_position = joint_position.item()
        
        if self.drive_type == "angular":
            joint_position = 180 * joint_position / np.pi

        percentage = (joint_position - self.lower)/(self.upper - self.lower) * 100

        
        # print("upper lower percentage", self.upper, self.lower, percentage)
        percentage = np.clip(percentage, 0, 100)

        return percentage 
    
    def compute_distance(self):
        return abs(self.compute_percentage() - self.initial_percentage)

    def set_joint(self, percentage):        
        joint_position = percentage / 100.0 *(self.upper-self.lower) + self.lower
        # if self.type == 'PhysicsPrismaticJoint':
        #     dof_pos = joint_position
        # else:
        #     dof_pos = math.radians(joint_position)
      
        self.joint_state_api.GetPositionAttr().Set(joint_position)


class JointChecker(BaseChecker):
    def __init__(self, checker_parameters: CheckerParameters, tolerance = 0.15) -> None:
        self.checker_parameters = checker_parameters
        self.tolerance = tolerance
    
    def pre_initialize(self, target_prim_path):
        super().__init__()
        self.target_joint = self.checker_parameters.target_joint

        self.init_value = self.checker_parameters.init_state
        self.target_value = self.checker_parameters.target_state
        self.target_prim_path = target_prim_path

        self.previous_percentage = None
        self.vel = None
        self.check_freq = 1

    def initialization_step(self):
        self.joint_checker = JointCheck(self.target_prim_path, self.target_joint)
        self.check_joint_direction()
        self.set_joint_at_start = True if self.init_value != -1 else False

        if hasattr(self.joint_checker, "articulation"):
            self.joint_checker.articulation.initialize()

        self.is_init = True
        self.create_task_callback()
            
    
    def check_joint_direction(self):
        """
        Check joint positive rotation to upper or negative rotation to lower
        """
        is_upper = abs(self.joint_checker.upper) > abs(self.joint_checker.lower)
        if not is_upper:
            # if is lower, reverse init_value and target value
            self.init_value = 1 - self.init_value if self.init_value != -1 else -1
            self.target_value = 1 - self.target_value

    def get_diff(self):
        percentage =  self.joint_checker.compute_percentage()
        return percentage/100  - self.target_value
        
    def start_checking(self):
        if self.is_init == False:
            return
        
        self.total_step += 1
        if self.total_step % self.check_freq == 0:
            if self.set_joint_at_start:
                self.joint_checker.set_joint(self.init_value*100)
                self.set_joint_at_start = False

            percentage =  self.joint_checker.compute_percentage()
        
            if self.previous_percentage is not None:
                self.vel  = abs(percentage - self.previous_percentage)

            if self.total_step % 120 == 0:
                print("current: {:.1f}; target: {:.1f}; delta percentage: {:.1f}:".format(percentage, self.target_value*100, self.target_value*100 - percentage) )
            
            if abs(percentage/100 - self.target_value) < self.tolerance and self.vel is not None and self.vel < 0.05:
                self.success_steps += self.check_freq
                self._on_success_hold()
            else:
                self._on_not_success()

            self.previous_percentage = percentage
            
            super().start_checking()


    def reset(self):
        if hasattr(self.joint_checker, "articulation"):
            self.joint_checker.articulation._physics_view = None
            self.joint_checker.articulation._is_initialized = False
            self.joint_checker.articulation._invalidation_callback = None
            
        super().reset()

        