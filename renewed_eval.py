import argparse
import json
import torch
import numpy as np
import os

from renewed_utils.data import load_data

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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render = args_cli.visualize


    #TODO: enable water tasks
    task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', #'pour_water', 'transfer_water'
        ]
    
    use_gt = args_cli.use_gt

    if use_gt[0]:
        if use_gt[1]:
            eval_setting = '2gt'
        else:
            eval_setting = '1gt'
    else:
        eval_setting = '0gt'
    log_path = os.path.join("output", f'eval_{eval_setting}_log.json')
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
