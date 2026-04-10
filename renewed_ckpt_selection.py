"""
Script for model selection. For example, run:
    Single-task:
        python ckpt_selection.py task=pickup_object model=peract lang_encoder=clip \
                                 mode=eval visualize=0
    Multi-task:
        python ckpt_selection.py task=multi model=peract lang_encoder=clip \
                                 mode=eval visualize=0
"""
import sys
import json
import logging
import numpy as np
import os
import shutil
from omegaconf import OmegaConf
import torch
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Eval checkpoints")
parser.add_argument('--task', type=str, default='pickup_object', help='Task to evaluate')
parser.add_argument('--model', type=str, default='peract', help='Model to use')
parser.add_argument('--lang_encoder', type=str, default='clip', help='Language encoder to use')
parser.add_argument('--mode', type=str, default='eval', help='Mode to use')
parser.add_argument('--visualize', type=int, default=0, help='Visualize the evaluation')
parser.add_argument("--cfg_path", type=str, default="./configs/default.yaml", help="Path to the config file.")
parser.add_argument("--checkpoint_dir", type=str, default="./data/model", help="Path to the checkpoint directory.")
parser.add_argument("--exp_dir", type=str, default="./output", help="Path to the experiment directory.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# suppress noisy warnings from rendering/semantics subsystems at launch time
sys.argv += [
    "--/log/channels/omni.hydra=error",
    "--/log/channels/isaacsim.core.utils.semantics=error",
    "--/log/channels/isaacsim.core.simulation_manager.plugin=error",
    "--/log/channels/usdrt.population.plugin=error",
    "--/log/channels/omni.physx.plugin=error",
    "--/log/channels/omni.usd.metrics.assembler.plugin=error",
    "--/log/channels/omni.physx.tensors.plugin=error",
]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dataset import InstructionEmbedding
from renewed_tasks import load_task
from local_utils.env import get_action

from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.simulation_cfg import PhysxCfg

import carb
logger = logging.getLogger(__name__)

from renewed_utils.data import load_data


def make_agent(model_name, cfg, device):
    lang_embed_cache = None
    if model_name == 'peract':
        from train_peract import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)
    elif 'bc_lang' in model_name:
        from train_bc_lang import create_agent, create_lang_encoder
        agent = create_agent(cfg, device=device)

        lang_encoder = create_lang_encoder(cfg, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)
    
    else:
        raise ValueError(f'{model_name} agent not supported')
    
    return agent, lang_embed_cache


def load_ckpt(model_name, agent, ckpt_path):
    if model_name == 'peract':
        agent.load_model(ckpt_path)
    elif 'bc_lang' in model_name:
        agent.load_weights(ckpt_path)
    else:
        raise ValueError(f'{model_name} agent not supported')
    
    logger.info(f"Loaded {model_name} from {ckpt_path}")
    return agent


def main():
    # load config
    cfg = OmegaConf.load(args_cli.cfg_path)

    device = 'cpu'# FIXME: fr debug 'cuda' if torch.cuda.is_available() else 'cpu'
    render = args_cli.visualize

    task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    non_fabric_tasks = {
        'open_drawer', 'close_drawer', 'open_cabinet', 'close_cabinet',
        'pour_water', 'transfer_water',
    }
    is_water_task = args_cli.task in ['pour_water', 'transfer_water']
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0, device=device,
        use_fabric=args_cli.task not in non_fabric_tasks,
        physx=PhysxCfg(enable_enhanced_determinism=True),
    )
    simulation_context = SimulationContext(sim_cfg)

    agent_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render = cfg.visualize

    offset = cfg.offset_bound
    use_gt = cfg.use_gt
    agent, lang_embed_cache = make_agent(args_cli.model, cfg, device=agent_device)
    
    if args_cli.task != 'multi':
        task_list = [args_cli.task]
    else:
        task_list = [
            'pickup_object', 'reorient_object', 'open_drawer', 'close_drawer',
            'open_cabinet', 'close_cabinet', 'pour_water', 'transfer_water'
        ]

    ckpts_list = [f for f in os.listdir(args_cli.checkpoint_dir) if f.endswith('pth')]

    for ckpt_name in ckpts_list:
        if 'best' in ckpt_name:
            logger.info('Best checkpoint already recognized')
            simulation_app.close()
            return 1
    # ckpts_list = sorted(ckpts_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    log_path = os.path.join(args_cli.exp_dir, 'select_log.json')
    """
    val log structure:
    {
        'ckpt_name': {
            'task': {
                # 'stats': {
                #     'fname': int (1, 0, -1)
                # },
                'score': float
            }
        }
    }
    """

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            val_log = json.load(f)
    else:
        val_log = {}

    for ckpt_name in ckpts_list:
        agent = load_ckpt(args_cli.model, agent, os.path.join(args_cli.checkpoint_dir, ckpt_name))
        if ckpt_name not in val_log:
            val_log[ckpt_name] = {}
        for task_name in task_list:
            if task_name not in val_log[ckpt_name]:
                val_log[ckpt_name][task_name] = {}
            elif 'score' in val_log[ckpt_name][task_name]:
                continue

            logger.info(f'Evaluating {ckpt_name} {task_name}')
            data, fnames = load_data(data_path=os.path.join("./data", task_name, 'val'))
            # data = load_data(data_path=os.path.join(cfg.data_root, task_name, 'val'))
            correct = 0
            total = 0
            env = None
            while len(data) > 0:
                anno = data.pop(0)
                gt_frames = anno['gt']
                robot_base = gt_frames[0]['robot_base']
                gt_actions = [
                    gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world'],
                    gt_frames[3]['position_rotation_world'] if 'water' not in task_name \
                    else (gt_frames[3]['position_rotation_world'][0], gt_frames[4]['position_rotation_world'][1])
                ]

                env, object_parameters, robot_parameters, scene_parameters = load_task(npz=anno, env=env)

                obs = env.reset(robot_parameters, scene_parameters, object_parameters, 
                                robot_base=robot_base, gt_actions=gt_actions)

                import ipdb; ipdb.set_trace()

                logger.info(f'Instruction: {gt_frames[0]["instruction"]}')
                logger.info('Ground truth action:')
                for gt_action, grip_open in zip(gt_actions, cfg.gripper_open[task_name]):
                    act_pos, act_rot = gt_action
                    act_rot = R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)
                    logger.info(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={grip_open}')

                try:
                    for i in range(2):
                        if use_gt[i]:
                            obs, suc = env.step(act_pos=None, act_rot=None, render=render, use_gt=True)
                        else:
                            act_pos, act_rot = get_action(
                                gt=obs, agent=agent, franka=env.robot, c_controller=env.c_controller, npz_file=anno, offset=offset, timestep=i,
                                device=agent_device, agent_type=cfg.model, obs_type=cfg.obs_type, lang_embed_cache=lang_embed_cache
                            )

                            logger.info(
                                f"Prediction action {i}: trans={act_pos}, orient(euler XYZ)={R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)}"
                            )

                            obs, suc = env.step(act_pos=act_pos, act_rot=act_rot, render=render, use_gt=False)

                        if suc == -1:
                            break
                
                except:
                    suc = -1

                env.stop()
                if suc == 1:
                    correct += 1
                total += 1
                log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
                logger.info(f'{log_str}\n')
            
            logger.info(f'{ckpt_name} {task_name}: {correct/total*100:.2f}\n\n')
            val_log[ckpt_name][task_name]['score'] = correct / total
            
            with open(log_path, 'w') as f:
                json.dump(val_log, f, indent=2)

    ckpt_scores = [np.mean([val_log[ckpt_name][task_name]['score'] for task_name in task_list]) for ckpt_name in ckpts_list]
    selected_idx = ckpt_scores.argmax()
    selected_name = ckpts_list[selected_idx]

    for ckpt_name in ckpts_list:
        if ckpt_name == selected_name:
            new_name = ckpt_name.split('_')
            new_name[-1] = 'best.pth'
            new_name = '_'.join(new_name)
            shutil.move(os.path.join(cfg.checkpoint_dir, ckpt_name), os.path.join(cfg.checkpoint_dir, new_name))
            logger.info(f'Select {selected_name} as best')
        # else:
        #     os.remove(os.path.join(cfg.checkpoint_dir, ckpt_name))
        #     logger.info(f'Remove {ckpt_name}')

    simulation_app.close()


if __name__ == '__main__':
    main()
