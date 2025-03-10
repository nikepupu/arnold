import os
import omni.ext
import omni.appwindow
import numpy as np
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction
import gzip
import json
from omni.isaac.core.utils.rotations import euler_angles_to_quat,quat_to_euler_angles, matrix_to_euler_angles, quat_to_rot_matrix

class DataRecorder():
    def __init__(self, robot_path, target_paths, frankabot, task_type):
        self.replayBuffer = []
        self.replayBufferObj = []
        self.record = False
        self.replay_start = False
        self.target_paths = target_paths
        self.robot_path = robot_path
        self.target_objs = [XFormPrim(t_path) for t_path in self.target_paths]

        self.traj_dir = None
        self._frankabot = frankabot
        self.legacy = False
        self.task_type = task_type
        self.checker = None
        self.ptcl_frame_skip = 20
        
        # New attribute to store the last non-None joint action values.
        self.last_joint_positions = None

        # Old-style buffer for CSV data
        self.buffer = {"robot": [], "object": [], "particles": [], "diffs": []}

    def get_replay_status(self):
        return self.replay_start

    def start_record(self, traj_dir, checker, language_instruction, raw_trajectory_path, object_id):
        self.replay_start = False
        self.record = True
        self.traj_dir = traj_dir
        self.checker = checker
        self.language_instruction = language_instruction
        self.raw_trajectory_path = raw_trajectory_path
        self.object_id = object_id

        # Reset old-style buffers
        self.buffer = {"robot": [], "object": [], "particle": []}

    def stop_record(self):
        self.record = False

    def save_buffer(self, success, abs_info=None):
        """
        1) Write old CSV/gzip files (as before),
        2) Write success/fail status to success.txt,
        3) If success == True, parse buffer and write new JSON in the desired format.
        """
        print("write:", self.traj_dir)
        

        # ----------------------------------------
        # NEW CODE: Build the JSON only if success
        # ----------------------------------------
        if success:
            trajectory_json = self._build_trajectory_json()
            
            # Get the base name without the .npz extension (e.g., "xxx")
            base_name = os.path.basename(self.traj_dir).replace('.npz', '')
            
            # Build a new directory path that includes the object_id as a subfolder
            new_dir = os.path.join(os.path.dirname(self.traj_dir), self.object_id)
            
            # Build the full output path with the new directory and file name
            out_path = os.path.join(new_dir, f"{base_name}_v2.json")
            
            # Ensure the new directory exists
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            with open(out_path, "w") as f:
                json.dump(trajectory_json, f, indent=2)
            print(f"[INFO] Wrote new JSON format to: {out_path}")
        
         # Reset old buffers
        self.buffer = {"robot": [], "object": [], "particle": []}

    def _build_trajectory_json(self):
        """
        Parse the data in self.buffer['robot'], self.buffer['object'] (already saved),
        and build a single-trajectory JSON in the format:
        {
          "franka": [
            {
              "actions": [...],
              "init_state": {...},
              "states": [...],
              "extra": None
            }
          ]
        }
        """
        num_steps = len(self.buffer["robot"])
        if num_steps == 0:
            return {"franka": []}

        traj_dict = {
            "franka": [
                {
                    "actions": [],
                    "init_state": {},
                    "states": [],
                    "extra": None
                }
            ]
        }
        ep = traj_dict["franka"][0]

        for i in range(num_steps):
            robot_line = self.buffer["robot"][i].strip()
            robot_data = eval(robot_line)
            action_dict_raw = eval(robot_data["actions"]) if robot_data.get("actions", None) else None

            object_line = self.buffer["object"][i].strip()
            object_data_list = eval(object_line)

            state_i = self._build_state_from_robot_and_objects(robot_data, object_data_list)
            action_i = self._build_action_from_raw(action_dict_raw)

            ep["states"].append(state_i)
            ep["actions"].append(action_i)

        ep["init_state"] = ep["states"][0]
        ep["init_state"]['Scene'] = {"pos": [0, 0, 0], "rot": euler_angles_to_quat(np.array([np.pi/2, 0, 0])).tolist() }
        ep["extra"] = {}
        ep['extra']['language_instruction'] = self.language_instruction
        ep['extra']['raw_trajectory_path'] = self.raw_trajectory_path
        return traj_dict

    def _build_state_from_robot_and_objects(self, robot_data, object_data_list):
        """
        Build a state dict with the robot and objects.
        """
        state_dict = {}

        # ------------------
        # Robot part ("franka")
        # ------------------
        joint_positions = robot_data.get("joint_pos", [])
        dof_pos_robot = {}
        for j, val in enumerate(joint_positions):
            # For indices 0-6, use panda_joint1...panda_joint7.
            if j == 7:
                key = "panda_finger_joint1"
            elif j == 8:
                key = "panda_finger_joint2"
            else:
                key = f"panda_joint{j+1}"
            dof_pos_robot[key] = float(val)
            if j in [7, 8]:
                dof_pos_robot[key] = float(val)/100.0

        state_dict["franka"] = {
            "pos": (self._frankabot.get_world_pose()[0]/100.0).tolist(),
            "rot": self._frankabot.get_world_pose()[1].tolist(),
            "dof_pos": dof_pos_robot
        }

        # ------------------
        # Objects
        # ------------------
        for obj_data in object_data_list:
            obj_path = obj_data["path"]
            obj_name = os.path.basename(obj_path)
            pos = obj_data["pos"]
            rot = obj_data["rot"]
            dof_pos_obj = {}
            if obj_data["joint"] is not None:
                dof_pos_obj[self.checker.target_joint] = float(obj_data["joint"])
                if self.task_type in ['open_drawer', 'close_drawer']:
                    dof_pos_obj[self.checker.target_joint] = float(obj_data["joint"])/100.0
            
            t_obj = self.target_objs[0]
            obj_trans, obj_rot = t_obj.get_world_pose()
            obj_trans = obj_trans/100.0
            obj_state = {
                "pos": [p/100.0 for p in pos] if pos else obj_trans.tolist(),
                "rot": rot if rot else obj_rot.tolist()
            }
            if dof_pos_obj:
                obj_state["dof_pos"] = dof_pos_obj

            state_dict[obj_name] = obj_state

        return state_dict

    def _build_action_from_raw(self, action_dict_raw):
        """
        Build an action dict from raw data.
        If a joint action value is missing, use the last non-None value.
        """
        if not action_dict_raw or "joint_positions" not in action_dict_raw:
            return {}

        joint_list = action_dict_raw["joint_positions"]
        dof_target_dict = {}
      
        for i, val in enumerate(joint_list):
            if val is None and self.last_joint_positions is not None:
                if i < len(self.last_joint_positions):
                    val = self.last_joint_positions[i]
            if val is None:
                continue

            if i == 7:
                key = "panda_finger_joint1"
            elif i == 8:
                key = "panda_finger_joint2"
            else:
                key = f"panda_joint{i+1}"
            
            try:
                dof_target_dict[key] = float(val)
            except:
                dof_target_dict[key] = val
            
            if i in [7, 8]:
                dof_target_dict[key] = float(val)/100.0

        if all(v is not None for v in joint_list):
            self.last_joint_positions = joint_list

        return {
            "dof_pos_target": dof_target_dict,
            "ee_pose_target": None
        }

    def delete_traj_folder(self):
        import shutil
        from pathlib import Path
        path = Path(self.traj_dir)
        path = path.parent.absolute()
        try:
            shutil.rmtree(path)
            print("replay failed, delete this traj: ", str(path))
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
    
    def record_data(self, robot_states, actions, time_step):
        self.obj_data = [{'pos': None, 'rot': None, 'joint': None, 'path': t_path} for t_path in self.target_paths]
        self.robot_data = {
            'joint_pos': robot_states['pos'].tolist(),
            'joint_vel': robot_states['vel'].tolist(),
            'actions': str(actions),
        }
        self.ptcl_data = {}
        
        if self.task_type in ["pickup_object", "reorient_object"]:
            self.record_obj_pose()
        elif self.task_type in ["open_drawer","open_cabinet", "close_drawer", "close_cabinet"]:
            self.record_obj_joint()
        elif self.task_type in ["pour_water", "transfer_water"]:
            self.record_obj_pose()
            self.record_ptcl(time_step)
        else:
            raise RuntimeError("data recording for %s not implemented" % self.task_type)

        if self.record:
            self.buffer['robot'].append(str(self.robot_data).replace("\n", ' ') + '\n')
            self.buffer['object'].append(str(self.obj_data).replace("\n", ' ') + '\n')
            self.buffer['particle'].append(str(self.ptcl_data).replace("\n", ' ') + '\n')

    def record_obj_pose(self):
        for idx, t_obj in enumerate(self.target_objs):
            obj_trans, obj_rot = t_obj.get_world_pose()
            self.obj_data[idx]["pos"] = obj_trans.tolist()
            self.obj_data[idx]["rot"] = obj_rot.tolist()

    def record_obj_joint(self):
        assert(self.checker is not None)
        for idx, t_path in enumerate(self.target_paths):
            if t_path == self.checker.target_prim_path:
                self.obj_data[idx]["joint"] = self.checker.joint_checker.compute_percentage()

    def record_ptcl(self, time_step):
        assert(self.checker is not None)
        if time_step % self.ptcl_frame_skip == 0:
            self.ptcl_data = self.checker.liquid_checker.get_all_particles()

    def start_replay(self, traj_dir, checker):
        self.traj_dir = traj_dir
        self.checker = checker
        if os.path.exists(os.path.join(self.traj_dir, 'record.csv')):
            with open(os.path.join(self.traj_dir, 'record.csv'), 'r') as file1:
                Lines = file1.readlines()
            try:
                self.replayBuffer = [eval(line) for line in Lines]
            except:
                self.replayBuffer = []
            self.legacy = True
        else:
            self.legacy = False
            self.replayBuffer = []
            self.replayBufferObj = []
            self.replayBufferPtcl = []

            def read_gzip_csv(file_name):
                try:
                    with gzip.open(os.path.join(self.traj_dir, f'{file_name}.csv.gz'), 'rt') as file:
                        return [eval(line) for line in file]
                except FileNotFoundError:
                    return []
            
            self.replayBuffer = read_gzip_csv('record_robot')
            self.replayBufferObj = read_gzip_csv('record_object')
            self.replayBufferPtcl = read_gzip_csv('record_particle')
            assert(len(self.replayBuffer) == len(self.replayBufferObj))
        self.replay_start = True

    def replay_data(self):
        taskDone = False
        if self.legacy == True:
            robot_data = self.replayBuffer.pop(0)
            action = robot_data
            if action is not None:
                if "joint_positions" in action and action["joint_positions"] is not None:
                    self.last_joint_positions = action["joint_positions"]
                    actions = ArticulationAction(joint_positions=action["joint_positions"])
                elif self.last_joint_positions is not None:
                    actions = ArticulationAction(joint_positions=self.last_joint_positions)
                else:
                    actions = None
                if actions is not None:
                    _articulation_controller = self._frankabot.get_articulation_controller()
                    _articulation_controller.apply_action(actions)
            if len(self.replayBuffer) == 0:
                taskDone = True
        else:
            if len(self.replayBuffer) > 0 and len(self.replayBufferObj) > 0:
                robot_data = self.replayBuffer.pop(0)
                obj_data = self.replayBufferObj.pop(0)
                
                ptcl_data = {}
                if self.replayBufferPtcl is not None and len(self.replayBufferPtcl) > 0:
                    ptcl_data = self.replayBufferPtcl.pop(0)
                
                if len(self.replayBuffer) % 1000 == 0:
                    print("len buffer: ", len(self.replayBuffer))

                if "joint_pos" in robot_data and robot_data["joint_pos"] is not None:
                    self._frankabot.set_joint_positions(robot_data["joint_pos"])
                    self._frankabot.set_joint_velocities(robot_data["joint_vel"])

                if 'actions' in robot_data:
                    action = eval(robot_data["actions"])
                    if action is None or action.get("joint_positions") is None:
                        joint_positions = self.last_joint_positions
                    else:
                        joint_positions = action["joint_positions"]
                        self.last_joint_positions = joint_positions
                    if joint_positions is not None:
                        actions = ArticulationAction(joint_positions=joint_positions)
                        _articulation_controller = self._frankabot.get_articulation_controller()
                        _articulation_controller.apply_action(actions)
                
                if self.task_type in ["pickup_object", "reorient_object"]:
                    self.replay_obj_pose(obj_data)
                elif self.task_type in ["open_drawer","open_cabinet", "close_drawer", "close_cabinet"]:
                    self.replay_obj_joint(obj_data)
                elif self.task_type in ["pour_water", "transfer_water"]:
                    self.replay_obj_pose(obj_data)
                    self.replay_ptcl(ptcl_data)
                else:
                    raise RuntimeError("data replay for %s not implemented" % self.task_type)
                    
            if len(self.replayBuffer) == 0 or len(self.replayBufferObj) == 0:
                taskDone = True

        return taskDone

    def replay_obj_pose(self, all_obj_data):
        for idx, obj_data in enumerate(all_obj_data):
            if "pos" in obj_data and "rot" in obj_data and \
               obj_data["pos"] is not None and obj_data["rot"] is not None:
                self.target_objs[idx].set_local_pose(
                    translation=np.array(obj_data["pos"]), 
                    orientation=np.array(obj_data["rot"])
                )

    def replay_obj_joint(self, all_obj_data):
        for idx, obj_data in enumerate(all_obj_data):
            if self.target_paths[idx] == self.checker.target_prim_path and \
               "joint" in obj_data and obj_data["joint"] is not None:
                self.checker.joint_checker.set_joint(np.array(obj_data["joint"]))

    def replay_ptcl(self, ptcl_data):
        if ptcl_data:
            self.checker.liquid_checker.set_all_particles(ptcl_data)
