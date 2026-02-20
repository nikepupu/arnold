import argparse
import json
import torch
import numpy as np
import os
import logging

from renewed_utils.data import load_data
from renewed_tasks import load_task

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Eval or Replay")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--mode", type=str, default="eval", help="eval or replay")
parser.add_argument("--visualize", action="store_true", default=False, help="Visualize the simulation.")
parser.add_argument("--use_gt", type=int, nargs=2, default=[1, 1], help="Use ground truth for action inputs.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim import SimulationCfg, SimulationContext
logger = logging.getLogger(__name__)


def main():
    device = 'cpu'# FIXME: fr debug 'cuda' if torch.cuda.is_available() else 'cpu'
    render = args_cli.visualize

    sim_cfg = SimulationCfg(dt=0.05, device=device)
    sim_context = SimulationContext(sim_cfg)


    #TODO: enable water tasks
    task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', #'pour_water', 'transfer_water'
        ]

    eval_splits = ['test', 'novel_object', 'novel_scene', 'novel_state', 'any_state']
    
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


    # TODO: write a forloop, for debug
    task = task_list[-1]
    eval_split = eval_splits[-1]
    logger.info(f'Evaluating {task} {eval_split}')

    data, fnames = load_data(data_path=os.path.join("./data", task, eval_split))

    # stats
    correct = 0
    total = 0
    stats = {}

    # TODO: write a while loop, for debug
    # while len(data) > 0:
    anno = data.pop(0)
    fname = fnames.pop(0)
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

    env, object_parameters, robot_parameters, scene_parameters = load_task(npz=anno)

    import ipdb; ipdb.set_trace()

    obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
        robot_base=robot_base, gt_actions=gt_actions)

    import ipdb; ipdb.set_trace()

    # Simulate
    while simulation_app.is_running():
        # perform step
        sim_context.step()

    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
