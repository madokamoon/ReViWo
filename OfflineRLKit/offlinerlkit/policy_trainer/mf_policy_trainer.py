import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy

import multiprocessing

from offlinerlkit.policy_trainer.mb_policy_trainer import worker, NCOLS

# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        logger: Logger,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gym_config = None,
        trainer_func = None,
        trainer_func_args = None,
        episode_worker: int = 1,
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        self.gym_config = gym_config
        self.gym_config['trainer'] = None
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
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)

                for k, v in loss.items():
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
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate_old(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            action = self.policy.select_action(obs.reshape(1,-1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def _evaluate(self, epoch) -> Dict[str, List[float]]:
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