import dmc
import glob
import numpy as np
from agents.cql_cds import CDSAgent
from agents.pbrl import PBRLAgent
import hydra
from pathlib import Path
import torch
import utils

def eval(agent, env, num_eval_episodes):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    while eval_until_episode(episode):
        time_step = env.reset()
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step=0, eval_mode=True)
            time_step = env.step(action)
            total_reward += time_step.reward
            step += 1
        episode += 1

    return total_reward / episode


### CDS

## walk
task_name = "walker_walk"
seed = 42
eval_env = dmc.make(task_name, seed=seed)
CDSagent = CDSAgent(name='cql_cds',
                 obs_shape=eval_env.observation_spec().shape, 
                 action_shape=eval_env.action_spec().shape,
                 device = 'cuda',
                 actor_lr = 1e-4,
                 critic_lr = 3e-4,
                 hidden_dim = 256,
                 critic_target_tau = 0.01,
                 nstep = 1,
                 batch_size = 1024,
                 use_tb = True,
                 alpha = 50,
                 n_samples = 3,
                 target_cql_penalty = 5.0,
                 use_critic_lagrange = False,
                 num_expl_steps=0)

# walk share
path = Path('result_cds/05-22-00-55-walker_walk-Share_walker_walk_walker_run-medium-cql_cds')
CDSagent.load(load_path=path)
print("CDS task:walker_walk dataset:share")
print(eval(env=eval_env, agent=CDSagent, num_eval_episodes=10))

# walk single
path = Path('result_cds/CDS-Single-walk')
CDSagent.load(load_path=path)
print("CDS task:walker_walk dataset:single")
print(eval(env=eval_env, agent=CDSagent, num_eval_episodes=10))

## run
task_name = "walker_run"
seed = 42
eval_env = dmc.make(task_name, seed=seed)
CDSagent = CDSAgent(name='cql_cds',
                 obs_shape=eval_env.observation_spec().shape, 
                 action_shape=eval_env.action_spec().shape,
                 device = 'cuda',
                 actor_lr = 1e-4,
                 critic_lr = 3e-4,
                 hidden_dim = 256,
                 critic_target_tau = 0.01,
                 nstep = 1,
                 batch_size = 1024,
                 use_tb = True,
                 alpha = 50,
                 n_samples = 3,
                 target_cql_penalty = 5.0,
                 use_critic_lagrange = False,
                 num_expl_steps=0)

# run share
path = Path('result_cds/05-26-17-41-walker_run-Share_walker_walk_walker_run-medium-cql_cds')
CDSagent.load(load_path=path)
print("CDS task:walker_run dataset:share")
print(eval(env=eval_env, agent=CDSagent, num_eval_episodes=10))

# run single
path = Path('result_cds/CDS-Single-run')
CDSagent.load(load_path=path)
print("CDS task:walker_run dataset:single")
print(eval(env=eval_env, agent=CDSagent, num_eval_episodes=10))


### UTDS

## walk
task_name = "walker_walk"
seed = 42
eval_env = dmc.make(task_name, seed=seed)
UTDSagent = PBRLAgent(name='pbrl',
                 obs_shape=eval_env.observation_spec().shape, 
                 action_shape=eval_env.action_spec().shape,
                 device = 'cuda',
                 lr=1e-4,
                 critic_target_tau=0.005,
                 actor_target_tau=0.005,
                 policy_freq=2,
                 use_tb=True,
                 hidden_dim=1024,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 batch_size=1024,
                 num_expl_steps=0,
                 num_random=3,
                 ucb_ratio_in=0.001,
                 ensemble=5,
                 ood_noise=0.01,
                 ucb_ratio_ood_init=3.0,
                 ucb_ratio_ood_min=0.1,
                 ood_decay_factor=0.99995,
                 share_ratio=1.5)

# walk share
path = Path('result_pbrl_share/walker_walk-Share_walker_walk_walker_run-medium-pbrl-05-23-09-04-32')
UTDSagent.load(load_path=path)
print("UTDS task:walker_walk dataset:share")
print(eval(env=eval_env, agent=UTDSagent, num_eval_episodes=10))

# walk single
path = Path('result_pbrl_share/walker_walk-Single_walker_walk_walker_run-medium-pbrl-05-24-17-16-20')
UTDSagent.load(load_path=path)
print("UTDS task:walker_walk dataset:single")
print(eval(env=eval_env, agent=UTDSagent, num_eval_episodes=10))

## run
task_name = "walker_run"
seed = 42
eval_env = dmc.make(task_name, seed=seed)
UTDSagent = PBRLAgent(name='pbrl',
                 obs_shape=eval_env.observation_spec().shape, 
                 action_shape=eval_env.action_spec().shape,
                 device = 'cuda',
                 lr=1e-4,
                 critic_target_tau=0.005,
                 actor_target_tau=0.005,
                 policy_freq=2,
                 use_tb=True,
                 hidden_dim=1024,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 batch_size=1024,
                 num_expl_steps=0,
                 num_random=3,
                 ucb_ratio_in=0.001,
                 ensemble=5,
                 ood_noise=0.01,
                 ucb_ratio_ood_init=3.0,
                 ucb_ratio_ood_min=0.1,
                 ood_decay_factor=0.99995,
                 share_ratio=1.5)

# run share
path = Path('result_pbrl_share/walker_run-Share_walker_walk_walker_run-medium-pbrl-05-25-19-17-24')
UTDSagent.load(load_path=path)
print("UTDS task:walker_run dataset:share")
print(eval(env=eval_env, agent=UTDSagent, num_eval_episodes=10))

# run single
path = Path('result_pbrl_share/walker_run-Single_walker_walk_walker_run-medium-pbrl-05-25-02-05-45')
UTDSagent.load(load_path=path)
print("UTDS task:walker_run dataset:single")
print(eval(env=eval_env, agent=UTDSagent, num_eval_episodes=10))


