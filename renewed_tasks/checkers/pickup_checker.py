import omni
import omni.usd
from isaacsim.core.prims import RigidPrim
from .base_checker import BaseChecker
from environment.parameters import CheckerParameters


class PickupChecker(BaseChecker):
    def __init__(self, checker_parameters: CheckerParameters, tolerance = 0.06) -> None:
        self.checker_parameters = checker_parameters
        self.tolerance = tolerance

    def pre_initialize(self, target_prim_path):
        super().__init__()

        self.target_prim_path = target_prim_path
        self.target_delta_y = self.checker_parameters.target_state/100.0
        self.previous_pos = None
        self.vel = None
        self.check_freq = 1

        self.target_prim = self.stage.GetPrimAtPath(self.target_prim_path)
        if not self.target_prim:
            raise Exception(f"Target prim must exist at path {self.target_prim_path}")

    def initialization_step(self):
        # Create the RigidPrim here (not in pre_initialize) because
        # the physics tensor views need time to absorb USD scene changes.
        # By this point the simulation has stepped enough for valid views.
        self.targetRigid = RigidPrim(prim_paths_expr=self.target_prim_path)
        self.targetRigid.initialize()
        pos, rot = self.targetRigid.get_world_poses(usd=False)
        self.target_prim_init_y = pos[0][1].item()
        self.is_init = True
        self.create_task_callback()
        
    def get_height(self):
        pos, rot = self.targetRigid.get_world_poses(usd=False)
        target_prim_current_y = pos[0][1].item()
        return target_prim_current_y

    def get_diff(self):
        pos, rot = self.targetRigid.get_world_poses(usd=False)
        target_prim_current_y = pos[0][1].item()
        need_delta_y = target_prim_current_y - (self.target_delta_y + self.target_prim_init_y)

        return need_delta_y
    
    def start_checking(self):
        if not self.is_init:
            return 
        
        self.total_step += 1
        if self.total_step % self.check_freq == 0:
            # mat = omni.usd.utils.get_world_transform_matrix(self.target_prim) 
            # target_prim_current_y = mat.ExtractTranslation()[1]
            
            pos, rot = self.targetRigid.get_world_poses(usd=False)
            # print("pos, rot", pos, rot)
            target_prim_current_y = pos[0][1].item()
            
            if self.previous_pos is not None:
                self.vel  = abs(target_prim_current_y - self.previous_pos)
            
            target_height = (self.target_delta_y + self.target_prim_init_y)
            need_delta_y = abs(target_prim_current_y - target_height)
            if self.total_step % self.print_every == 0:
                print(self.total_step, self.target_prim_path, "target height %s current height %s" %(target_height, target_prim_current_y))
                print("tolerance", self.tolerance, "vel", self.vel, "need_delta_y", need_delta_y)

            # success condition
            if  need_delta_y < self.tolerance and self.vel is not None and self.vel < 0.1 :
                # import ipdb; ipdb.set_trace()
                self.success_steps += self.check_freq
                self._on_success_hold()
            else:
                # self.success = False
                self._on_not_success()
            self.previous_pos = target_prim_current_y
            super().start_checking()
