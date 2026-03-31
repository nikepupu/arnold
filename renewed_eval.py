import argparse
import json
import torch
import numpy as np
import os
import logging
import yaml
from scipy.spatial.transform import Rotation as R

from renewed_utils.data import load_data
from renewed_tasks import load_task

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Eval or Replay")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--mode", type=str, default="eval", help="eval or replay")
parser.add_argument("--visualize", action="store_true", default=False, help="Visualize the simulation.")
parser.add_argument("--use_gt", type=int, nargs=2, default=[1, 1], help="Use ground truth for action inputs.")
parser.add_argument("--record", action="store_true", default=False, help="Record trajectories.")
parser.add_argument("--cfg_path", type=str, default="./configs/default.yaml", help="Path to the config file.")
parser.add_argument("--start_episode", type=int, default=0, help="Episode number to start from (0-indexed).")
parser.add_argument("--split", type=str, default="test", help="Data split to evaluate (e.g. train, test, val).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# Disable multi-GPU to avoid Hydra scene delegate segfaults with dual-GPU ICD configs
args_cli.multi_gpu = False
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim import SimulationCfg, SimulationContext
logger = logging.getLogger(__name__)


def main():
    # load config
    with open(os.path.join(args_cli.cfg_path), 'r') as f:
        cfg = yaml.safe_load(f)

    device = 'cpu'# FIXME: fr debug 'cuda' if torch.cuda.is_available() else 'cpu'
    render = args_cli.visualize

    task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    is_water_task = args_cli.task in ['pour_water', 'transfer_water']
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0, device=device,
        use_fabric=not is_water_task,
    )
    simulation_context = SimulationContext(sim_cfg)

    task = args_cli.task
    eval_split = args_cli.split
    assert task in task_list, f"Task {task} not in {task_list}"
    logger.info(f'Evaluating {task} {eval_split}')


    
    use_gt = args_cli.use_gt

    if use_gt[0]:
        if use_gt[1]:
            eval_setting = '2gt'
        else:
            eval_setting = '1gt'
    else:
        eval_setting = '0gt'
    
    log_path = os.path.join("output", f'eval_{eval_setting}_log.json')
    """
    eval log structure:
    {
        'task_name': {
            'split': {
                'stats': {
                    'fname': int (1, 0, -1)
                },
                'score': float
            }
        }
    }
    """

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            eval_log = json.load(f)
    else:
        eval_log = {}




    data, fnames = load_data(data_path=os.path.join("./data", task, eval_split))

    # stats
    correct = 0
    total = 0
    stats = {}
    env = None

    while len(data) > 0:
        anno = data.pop(0)
        fname = fnames.pop(0)

        if total < args_cli.start_episode:
            print(f'=== Episode {total}: {fname} [SKIPPED] ===', flush=True)
            total += 1
            continue

        gt_frames = anno['gt']
        robot_base = gt_frames[0]['robot_base']

        gt_actions = [gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world']]
        if gt_frames[3]['position_rotation_world'] is not None:
            gt_actions.append(
                gt_frames[3]['position_rotation_world'] if 'water' not in task \
                else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
            )
        else:
            gt_actions.append(None)

        if use_gt[0]:
            assert gt_actions[0] is not None and gt_actions[1] is not None, "Use first gt action but it is missing"
        if use_gt[1]:
            assert gt_actions[2] is not None, "Use second gt action but it is missing"

        print(f'=== Episode {total}: {fname} ===', flush=True)
        env, object_parameters, robot_parameters, scene_parameters = load_task(npz=anno, env=env)

        obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
            robot_base=robot_base, gt_actions=gt_actions)

        logger.info(f'Instruction: {gt_frames[0]["instruction"]}')
        logger.info('Ground truth action:')
        for gt_action, grip_open in zip(gt_actions, cfg['gripper_open'][task]):
            if gt_action is None:
                continue
            act_pos, act_rot = gt_action
            act_rot = R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)
            logger.info(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={grip_open}')

        # TODO: check recording
        if args_cli.record:
            env.recorder.start_record(
                traj_dir=os.path.join(cfg.exp_dir, f'traj_{eval_setting}', os.path.split(fname)[-1]),
                checker=env.checker,
            )

        for i in range(2):
            if use_gt[i]:
                obs, suc = env.step(act_pos=None, act_rot=None, render=render, use_gt=True)
            else:
                act_pos, act_rot = get_action(
                    gt=obs, agent=agent, franka=env.robot, c_controller=env.c_controller, npz_file=anno, offset=offset, timestep=i,
                    device=device, agent_type=cfg.model, obs_type=cfg.obs_type, lang_embed_cache=lang_embed_cache
                )

                logger.info(
                    f"Prediction action {i}: trans={act_pos}, orient(euler XYZ)={R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)}"
                )

                obs, suc = env.step(act_pos=act_pos, act_rot=act_rot, render=render, use_gt=False)

            if suc == -1:
                break
        
        env.stop()
        if suc == 1:
            correct += 1
            print(f'  SUCCESS', flush=True)
        else:
            print(f'  FAILED (suc={suc})', flush=True)
        total += 1
        rate = correct / total
        print(f'  Rate so far: {correct}/{total} = {rate:.4f}', flush=True)
        stats[fname] = suc

    score = correct / total if total > 0 else 0.0
    print(f'RESULT {task} {eval_split} {correct}/{total} {score:.4f}', flush=True)

    if task not in eval_log:
        eval_log[task] = {}
    eval_log[task][eval_split] = {
        'stats': stats,
        'score': score,
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(eval_log, f, indent=2)

    simulation_app.close()

    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
