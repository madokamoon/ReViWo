import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

import multiprocessing
import warnings

NCOLS = 130


# model-based policy trainer
class MBPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        real_ratio: float = 0.05,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        dynamics_update_freq: int = 0,
        gym_config = None,
        trainer_func = None,
        trainer_func_args = None,
        episode_worker: int = 1,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger

        self._rollout_freq, self._rollout_batch_size, \
            self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.gym_config = gym_config
        # self.gym_config['trainer'] = None
        self.trainer_func = trainer_func
        self.trainer_func_args = trainer_func_args

        self.episode_worker = episode_worker

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}", ncols=NCOLS)
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log(
                        "num rollout transitions: {}, reward mean: {:.4f}".\
                            format(rollout_info["num_transitions"], rollout_info["reward_mean"])
                    )
                    for _key, _value in rollout_info.items():
                        self.logger.logkv_mean("rollout_info/"+_key, _value)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size
                real_batch = self.real_buffer.sample(batch_size=real_sample_size)
                fake_batch = self.fake_buffer.sample(batch_size=fake_sample_size)
                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                
                # update the dynamics if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps+1)%self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics(self.real_buffer)
                    for k, v in dynamics_update_info.items():
                        self.logger.logkv_mean(k, v)
                
                num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.logger.logkv("train/lr", self.lr_scheduler.get_last_lr()[0])
            
            # evaluate current policy
            eval_info = self._evaluate(epoch=e)
            succ_rate = eval_info['eval/succ_rate']
            self.logger.logkv("eval/succ_rate", succ_rate)
            ep_reward_d_mean = np.mean(eval_info["eval/episode_reward_d"])
            self.logger.logkv("eval/dynamic_episode_reward", ep_reward_d_mean)
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            # norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            # norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean)
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std)
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs(exclude=["dynamics_training_progress"])
        
            # save checkpoint
            # torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        # torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        # self.policy.dynamics.save(self.logger.model_dir)
        self.logger.close()
    
        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self, epoch: int) -> Dict[str, List[float]]:
        self.policy.eval()
                            
        episode_worker = self.episode_worker  # how many episodes each worker should handle
        assert (self._eval_episodes % episode_worker) == 0
        num_process = self._eval_episodes // episode_worker  # num of processes
        results = []
        with multiprocessing.Pool(processes=num_process) as pool:            
            for process_id in range(num_process):
                result = pool.apply_async(worker, (
                    self.policy,
                    self.trainer_func,
                    self.trainer_func_args,
                    self.gym_config,
                    episode_worker,
                    process_id,
                    epoch,
                    ))
                results.append(result)

            pool.close()
            pool.join()
        
        eval_ep_info_buffer = []
        num_episodes, succ_num = 0, 0
        for idx in range(num_process):
            eval_ep_info_buffer_temp, num_episodes_temp, succ_num_temp = results[idx].get()
            eval_ep_info_buffer.extend(eval_ep_info_buffer_temp)
            num_episodes += num_episodes_temp
            succ_num += succ_num_temp
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
            "eval/succ_rate" : succ_num / num_episodes,
            "eval/episode_reward_d": [ep_info["episode_reward_d"] for ep_info in eval_ep_info_buffer],
        }


import gymnasium as gym1
from copy import deepcopy
def worker(policy: BasePolicy, trainer_func, trainer_func_args, gym_config, episode_worker: int = 1, process_id: int = 0, epoch: int = 0):
    # print(f'process: {process_id}, start')

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # get worker specific env
    os.environ['MASTER_PORT'] = f'{12346+process_id}'
    if gym_config['tokenizer'] is None:
        gym_config = deepcopy(gym_config)
        trainer_local = trainer_func(**trainer_func_args)
        gym_config['tokenizer'] = trainer_local
    if gym_config['save_demo_path'] is not None:
        gym_config['save_demo_path'] = gym_config['save_demo_path'].format(epoch=epoch)
    env = gym1.make(**(gym_config)).env.env
    # env = gym.make(**(gym_config))
    env.seed(process_id)

    obs = env.reset()

    if process_id == 0:
        pbar = tqdm(total=gym_config['max_eps_step'] * episode_worker, desc='process_0_len', ncols=NCOLS)

    num_episodes_temp, succ_num_temp = 0, 0
    episode_reward, episode_length = 0, 0
    episode_reward_d = 0
    eval_ep_info_buffer_temp = []
    while num_episodes_temp < episode_worker:
        action = policy.select_action(obs.reshape(1, -1), deterministic=True)
        next_obs, reward, terminal, info = env.step(action.flatten())
        if hasattr(policy, 'dynamics'):
            _, reward_d, _, _ = policy.dynamics.step(obs=obs, action=action.flatten())
            episode_reward_d += reward_d
        episode_reward += reward
        episode_length += 1

        obs = next_obs

        # print(f'process: {process_id}, len: {episode_length}')
        if process_id == 0:
            pbar.update()

        if terminal:
            eval_ep_info_buffer_temp.append(
                {"episode_reward": episode_reward, "episode_length": episode_length, "episode_reward_d": episode_reward_d}
            )
            num_episodes_temp += 1
            episode_reward, episode_length = 0, 0
            episode_reward_d = 0
            env.seed(process_id + num_episodes_temp * 100)
            obs = env.reset()

            if info['is_success']:
                succ_num_temp += 1

            if process_id == 0:
                pbar.set_postfix({'num_episodes': num_episodes_temp})
    
    return eval_ep_info_buffer_temp, num_episodes_temp, succ_num_temp
