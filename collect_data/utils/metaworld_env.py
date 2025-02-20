"""
Simple wrapper for registering metaworld enviornments
properly with gym.
"""
import gymnasium as gym
import numpy as np
import torch
import os
import copy
from dataclasses import dataclass

try:
    import metaworld
    from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
except:
    pass

SUCCESS_REWARD = 10  
from PIL import Image
import imageio

import random

def count_direct_files_in_directory(directory_path):
    with os.scandir(directory_path) as entries:
        file_count = sum(1 for entry in entries if entry.is_file())
    return file_count

class SawyerEnv4CLIP(gym.Env):
    def __init__(self, 
                 env_name, 
                 max_eps_step: int=128, 
                 use_reward_shaping: bool=False, 
                 use_camera: bool=False, 
                 seed: bool=True, 
                 save_demo_path: str=None, 
                 n_eval_episodes: int=100, 
                 collect_data: bool=False,
                 is_gymnasium: bool=True):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
        
        self.is_gymnasium = is_gymnasium
        self.env_name = env_name
        self.use_camera = use_camera
        if self.use_camera:
            os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
            os.environ['MUJOCO_GL'] = 'osmesa'
            self._env = ALL_V2_ENVIRONMENTS[env_name](render_mode='rgb_array')        
            # self.set_render_config()  
            self.save_demo_path = save_demo_path
            self.save_demo_freq = n_eval_episodes
            self.all_eps = 0
        else:
            self._env = ALL_V2_ENVIRONMENTS[env_name]()
        self._env._freeze_rand_vec = False
        self._env._set_task_called = True
        self._seed = seed
        if self._seed:
            self._env.seed(0)  # Seed it at zero for now.

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._max_episode_steps = max_eps_step
        self.use_reward_shaping = use_reward_shaping
        self.collect_data = collect_data
        if self.use_reward_shaping:
            self.prev_reward = None
            self.scale = 1.0
        
    def set_render_config(self, ):
        DEFAULT_CAMERA_CONFIG = {
        "distance": 1.5,
        "azimuth": 135,
        "elevation": -20.0,
        "lookat": np.array([0, 0.5, 0.0])
        }
        self._env.mujoco_renderer = MujocoRenderer(self._env.model, self._env.data, DEFAULT_CAMERA_CONFIG)
        self._env.mujoco_renderer.width = 480
        self._env.mujoco_renderer.height = 480

    def seed(self, seed=None):
        if self._seed:
            self._env.seed(seed)

    def evaluate_state(self, state, action):
        return self._env.evaluate_state(state, action)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        if self._episode_steps == self._max_episode_steps:
            terminated = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        info['is_success'] = bool(info['success'])
        terminated = terminated or info['is_success']
        self.current_obs = obs

        if self.use_reward_shaping:
            reward_info = {
                'reward': reward,
                'done': terminated or truncated,
                'info': info,
            }
            reward = self.reward_shaping(reward_info=reward_info)
            
        if self.use_camera:
            fig_array = self._env.render()
            self.fig_arrays.append(fig_array)
            
            if (terminated or truncated) and self.save_demo_path is not None:
                ### save the demo to the given path
                for _ in range(9):
                    self.fig_arrays.append(fig_array)
                fig_list = [Image.fromarray(item) for item in self.fig_arrays]
                all_demo_num = count_direct_files_in_directory(self.save_demo_path)
                test_time = all_demo_num // self.save_demo_freq
                test_num = all_demo_num - test_time * self.save_demo_freq
                imageio.mimsave(self.save_demo_path + f'/{test_time}_{test_num}_output.gif', fig_list[::4], fps=10)
                self.fig_arrays = self.fig_arrays[:-9]
                
            if (terminated or truncated) and self.collect_data:
                if len(self.fig_arrays) > self._max_episode_steps:
                    self.fig_arrays = self.fig_arrays[:self._max_episode_steps]
                info['fig_arrays'] = self.fig_arrays[:]
        
        if self.is_gymnasium:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated or truncated, info

    def set_state(self, state):
        qpos, qvel = state[: self._env.model.nq], state[self._env.model.nq :]
        self._env.set_state(qpos, qvel)

    def reset(self, **kwargs):
        self._episode_steps = 0
        obs, info = self._env.reset(**kwargs)
        self.current_obs = obs
        if self.use_reward_shaping:
            self.prev_reward = 0.0
        if self.use_camera:
            fig_array = self._env.render()
            self.fig_arrays = [fig_array]
            self.all_eps += 1
        if self.is_gymnasium:
            return obs, info
        else:
            return obs
    
    def reward_shaping(self, reward_info: dict):
        reward = reward_info['reward']
        done = reward_info['done']
        info = reward_info['info']

        curr_reward = reward
        reward = (curr_reward - self.prev_reward) * self.scale
        self.prev_reward = curr_reward
        if done:
            if info['is_success']:
                reward = SUCCESS_REWARD

        return reward

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def get_normalized_score(self, score):
        return score
    
    
class SawyerEnv4MultiView(SawyerEnv4CLIP):
    def __init__(self, 
                 env_name, 
                 max_eps_step: int=128, 
                 use_reward_shaping: bool=False, 
                 use_camera: bool=False, 
                 seed: bool=True, 
                 save_demo_path: str=None, 
                 n_eval_episodes: int=100,  
                 collect_data: bool=False,
                 azimuths: list=[0, 45, 67.5, 112.5, 135, 180, 225, 315]):
        super().__init__(env_name, max_eps_step, use_reward_shaping, use_camera, seed, save_demo_path, n_eval_episodes, collect_data)
        
        self.azimuths = azimuths
        self.fig_arrays_dict = {}
        for azimuth in self.azimuths:
            self.fig_arrays_dict[str(azimuth)] = []
    
    
    def set_render_config(self, azimuth):
        DEFAULT_CAMERA_CONFIG = {
        "distance": 1.5,
        "azimuth": azimuth,
        "elevation": -20.0,
        "lookat": np.array([0, 0.5, 0.0])
        }
        self._env.mujoco_renderer = MujocoRenderer(self._env.model, self._env.data, DEFAULT_CAMERA_CONFIG)
        
    def render(self, ):
        # print(f"Step {self._episode_steps} start rendering!")
        for azimuth in self.azimuths:
            self.set_render_config(azimuth=azimuth)
            fig = Image.fromarray(self._env.render()).resize((128, 128))
            self.fig_arrays_dict[str(azimuth)].append(np.array(fig, dtype=np.uint8))
        
    def reset(self, **kwargs):
        self.fig_arrays_dict = {}
        for azimuth in self.azimuths:
            self.fig_arrays_dict[str(azimuth)] = []
        self._episode_steps = 0
        obs, info = self._env.reset(**kwargs)
        self.current_obs = obs
        if self.use_reward_shaping:
            self.prev_reward = 0.0
        if self.use_camera:
            self.render()
            self.all_eps += 1
        return obs, info
    
    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        if self._episode_steps == self._max_episode_steps:
            terminated = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        info['is_success'] = bool(info['success'])
        terminated = terminated or info['is_success']
        self.current_obs = obs

        if self.use_reward_shaping:
            reward_info = {
                'reward': reward,
                'done': terminated or truncated,
                'info': info,
            }
            reward = self.reward_shaping(reward_info=reward_info)
            
        if self.use_camera:
            self.render()
            
            if (terminated or truncated) and self.save_demo_path is not None:
                ### save the demo to the given path
                for azimuth in self.azimuths:
                    fig_array = self.fig_arrays_dict[str(azimuth)][-1]
                    for _ in range(9):
                        self.fig_arrays_dict[str(azimuth)].append(fig_array)
                    fig_list = [Image.fromarray(item) for item in self.fig_arrays_dict[str(azimuth)]]

                    version = 0
                    while os.path.exists(self.save_demo_path + f'/{self.env_name}_{azimuth}_{version}.gif'):
                        version += 1
                    
                    imageio.mimsave(self.save_demo_path + f'/{self.env_name}_{azimuth}_{version}.gif', fig_list, fps=20)
                    self.fig_arrays_dict[str(azimuth)] = self.fig_arrays_dict[str(azimuth)][:-9]
            
            if (terminated or truncated) and self.collect_data:
                fig_dict = {}
                for azimuth in self.azimuths:
                    if len(self.fig_arrays_dict[str(azimuth)]) > self._max_episode_steps:
                        self.fig_arrays_dict[str(azimuth)] = self.fig_arrays_dict[str(azimuth)][:self._max_episode_steps]
                    fig_dict[str(azimuth)] = np.array(self.fig_arrays_dict[str(azimuth)], dtype=np.uint8)
                info['fig_arrays'] = copy.deepcopy(fig_dict)
        
        return obs, reward, terminated, truncated, info
            
@dataclass
class CameraConfig:
    distance = 1.5
    azimuth = 0
    elevation = -20.0
    lookat = np.array([0, 0.5, 0.0])


class SawyerEnv4MultiCameraConfig(SawyerEnv4CLIP):
    def __init__(self, 
                 env_name, 
                 camera_ids, 
                 camera_configs, 
                 max_eps_step: int=128, 
                 use_reward_shaping: bool=False, 
                 use_camera: bool=False, 
                 seed: bool=True, 
                 save_demo_path: str=None, 
                 n_eval_episodes: int=100, 
                 collect_data: bool=False):
        super().__init__(env_name, max_eps_step, use_reward_shaping, use_camera, seed, save_demo_path, n_eval_episodes, collect_data)
        
        self.camera_configs = camera_configs
        self.camera_ids = camera_ids
        self.fig_arrays_dict = {}
        for camera_id in self.camera_ids:
            self.fig_arrays_dict[camera_id] = []
    
    def set_camera_ids_configs(self, camera_ids, camera_configs):
        self.camera_ids = camera_ids
        self.camera_configs = camera_configs
    
    def set_render_config(self, camera_config: CameraConfig):
        DEFAULT_CAMERA_CONFIG = {
        "distance": camera_config.distance,
        "azimuth": camera_config.azimuth,
        "elevation": camera_config.elevation,
        "lookat": camera_config.lookat
        }
        self._env.mujoco_renderer = MujocoRenderer(self._env.model, self._env.data, DEFAULT_CAMERA_CONFIG)
        self._env.mujoco_renderer.width = 480
        self._env.mujoco_renderer.height = 480
        
    def render(self, ):
        # print(f"Step {self._episode_steps} start rendering!")
        for i, camera_config in enumerate(self.camera_configs):
            self.set_render_config(camera_config=camera_config)
            fig = Image.fromarray(self._env.render()).resize((128, 128))
            self.fig_arrays_dict[self.camera_ids[i]].append(np.array(fig, dtype=np.uint8))
        
    def reset(self, **kwargs):
        self.fig_arrays_dict = {}
        for camera_id in self.camera_ids:
            self.fig_arrays_dict[camera_id] = []
        self._episode_steps = 0
        obs, info = self._env.reset(**kwargs)
        self.current_obs = obs
        if self.use_reward_shaping:
            self.prev_reward = 0.0
        if self.use_camera:
            self.render()
            self.all_eps += 1
        return obs, info

    def get_arrays(self, ):
        fig_dict = {}
        for camera_id in self.camera_ids:
            if len(self.fig_arrays_dict[camera_id]) > self._max_episode_steps:
                self.fig_arrays_dict[camera_id] = self.fig_arrays_dict[camera_id][:self._max_episode_steps]
            fig_dict[camera_id] = np.array(self.fig_arrays_dict[camera_id], dtype=np.uint8)
        return fig_dict

    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        if self._episode_steps == self._max_episode_steps:
            terminated = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        info['is_success'] = bool(info['success'])
        terminated = terminated or info['is_success']
        self.current_obs = obs

        if self.use_reward_shaping:
            reward_info = {
                'reward': reward,
                'done': terminated or truncated,
                'info': info,
            }
            reward = self.reward_shaping(reward_info=reward_info)
            
        if self.use_camera and self._episode_steps % 8 == 0:
            self.render()
            
            if (terminated or truncated) and self.save_demo_path is not None:
                ### save the demo to the given path
                for camera_id in self.camera_ids:
                    fig_array = self.fig_arrays_dict[camera_id][-1]
                    for _ in range(9):
                        self.fig_arrays_dict[camera_id].append(fig_array)
                    fig_list = [Image.fromarray(item) for item in self.fig_arrays_dict[camera_id]]

                    version = 0
                    while os.path.exists(self.save_demo_path + f'/{self.env_name}_{camera_id}_{version}.gif'):
                        version += 1
                    
                    imageio.mimsave(self.save_demo_path + f'/{self.env_name}_{camera_id}_{version}.gif', fig_list, fps=20)
                    self.fig_arrays_dict[camera_id] = self.fig_arrays_dict[camera_id][:-9]
            
        if (terminated or truncated) and self.collect_data:
            fig_dict = {}
            for camera_id in self.camera_ids:
                if len(self.fig_arrays_dict[camera_id]) > self._max_episode_steps:
                    self.fig_arrays_dict[camera_id] = self.fig_arrays_dict[camera_id][:self._max_episode_steps]
                fig_dict[camera_id] = np.array(self.fig_arrays_dict[camera_id], dtype=np.uint8)
            info['fig_arrays'] = copy.deepcopy(fig_dict)
        
        return obs, reward, terminated, truncated, info
    
class SawyerEnv4SingleView(SawyerEnv4CLIP):
    def __init__(self, 
                 env_name, 
                 camera_id, 
                 camera_config, 
                 max_eps_step: int=128, 
                 use_reward_shaping: bool=False, 
                 use_camera: bool=False, 
                 seed: bool=True, 
                 save_demo_path: str=None, 
                 n_eval_episodes: int=100, 
                 collect_data: bool=False, 
                 is_gymnasium: bool=True):
        super().__init__(env_name, max_eps_step, use_reward_shaping, use_camera, seed, save_demo_path, n_eval_episodes, collect_data, is_gymnasium)
        
        self.camera_config = camera_config
        self.camera_id = camera_id
        self.fig_arrays = []
    
    def set_camera_ids_configs(self, camera_id, camera_config):
        self.camera_id = camera_id
        self.camera_config = camera_config
    
    def set_render_config(self, camera_config: CameraConfig):
        DEFAULT_CAMERA_CONFIG = {
        "distance": camera_config.distance,
        "azimuth": camera_config.azimuth,
        "elevation": camera_config.elevation,
        "lookat": camera_config.lookat
        }
        self._env.mujoco_renderer = MujocoRenderer(self._env.model, self._env.data, DEFAULT_CAMERA_CONFIG)
        self._env.mujoco_renderer.width = 480
        self._env.mujoco_renderer.height = 480
        
    def render(self, ):
        # print(f"Step {self._episode_steps} start rendering!")
        self.set_render_config(camera_config=self.camera_config)
        fig = Image.fromarray(self._env.render()).resize((128, 128))
        self.fig_arrays.append(np.array(fig, dtype=np.uint8))
        
    def reset(self, **kwargs):
        self.fig_arrays = []
        self._episode_steps = 0
        obs, info = self._env.reset(**kwargs)
        self.current_obs = obs
        if self.use_reward_shaping:
            self.prev_reward = 0.0
        if self.use_camera:
            self.render()
            self.all_eps += 1
        if self.is_gymnasium:
            return obs, info
        else:
            return obs
        
    def get_arrays(self, ):
        if len(self.fig_arrays) > self._max_episode_steps:
            self.fig_arrays = self.fig_arrays[:self._max_episode_steps]
        fig_arrays = np.array(self.fig_arrays, dtype=np.uint8)
        return fig_arrays
    
    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        if self._episode_steps == self._max_episode_steps:
            terminated = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        info['is_success'] = bool(info['success'])
        terminated = terminated or info['is_success']
        self.current_obs = obs

        if self.use_reward_shaping:
            reward_info = {
                'reward': reward,
                'done': terminated or truncated,
                'info': info,
            }
            reward = self.reward_shaping(reward_info=reward_info)
            
        if self.use_camera:
            self.render()
            
            if (terminated or truncated) and self.save_demo_path is not None:
                ### save the demo to the given path
                fig_array = self.fig_arrays[-1]
                for _ in range(9):
                    self.fig_arrays.append(fig_array)
                fig_list = [Image.fromarray(item) for item in self.fig_arrays]

                version = 0
                while os.path.exists(self.save_demo_path + f'/{self.env_name}_{version}.gif'):
                    version += 1
                
                imageio.mimsave(self.save_demo_path + f'/{self.env_name}_{version}.gif', fig_list, fps=20)
                self.fig_arrays = self.fig_arrays[:-9]
            
            if (terminated or truncated) and self.collect_data:
                if len(self.fig_arrays) > self._max_episode_steps:
                    self.fig_arrays = self.fig_arrays[:self._max_episode_steps]
                fig_arrays = np.array(self.fig_arrays, dtype=np.uint8)
                info['fig_arrays'] = fig_arrays
                info['camera_id'] = self.camera_id
                info['camera_config'] = self.camera_config
                
        if self.is_gymnasium:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated or truncated, info

from common.tokenizer_wrappers import MultiViewTokenizer
class SawyerEnv4SingleViewWithLatentEmbedding(SawyerEnv4CLIP):
    def __init__(self, 
                 env_name, 
                 camera_id, 
                 camera_config, 
                 tokenizer: MultiViewTokenizer, 
                 max_eps_step: int=128, 
                 use_reward_shaping: bool=False, 
                 use_camera: bool=True, 
                 seed: bool=True, 
                 save_demo_path: str=None, 
                 n_eval_episodes: int=100, 
                 collect_data: bool=False, 
                 is_gymnasium: bool=True,
                 do_shake: bool=False,
                 ):
        self.tokenizer = tokenizer
        
        super().__init__(env_name, max_eps_step, use_reward_shaping, use_camera, seed, save_demo_path, n_eval_episodes, collect_data, is_gymnasium)
        
        self.camera_config = camera_config
        self.camera_id = camera_id
        self.fig_arrays = []

        self.do_shake = do_shake
        if self.do_shake:
            self.current_azimuth = camera_config.azimuth
            self.shake_size = 0
            self.shake_step = 0

    def set_camera_ids_configs(self, camera_id, camera_config):
        self.camera_id = camera_id
        self.camera_config = camera_config
    
    def set_render_config(self, camera_config: CameraConfig):
        DEFAULT_CAMERA_CONFIG = {
        "distance": camera_config.distance,
        "azimuth": camera_config.azimuth,
        "elevation": camera_config.elevation,
        "lookat": camera_config.lookat
        }
        self._env.mujoco_renderer = MujocoRenderer(self._env.model, self._env.data, DEFAULT_CAMERA_CONFIG)
        self._env.mujoco_renderer.width = 480
        self._env.mujoco_renderer.height = 480
        
    def render(self, camera_config=None):
        # print(f"Step {self._episode_steps} start rendering!")
        if camera_config is None:
            camera_config = self.camera_config
        self.set_render_config(camera_config=camera_config)
        fig = Image.fromarray(self._env.render()).resize((128, 128))
        self.fig_arrays.append(np.array(fig, dtype=np.uint8))
        
    def reset(self, **kwargs):
        self.fig_arrays = []
        self._episode_steps = 0
        obs, info = self._env.reset(**kwargs)
        self.current_obs = obs
        if self.use_reward_shaping:
            self.prev_reward = 0.0
        if self.use_camera:
            self.render()
            self.all_eps += 1
            obs = self.tokenizer.get_latent_embedding(self.fig_arrays[-1][np.newaxis, ...])
        
        if self.is_gymnasium:
            return obs, info
        else:
            return obs
    
    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        if self._episode_steps == self._max_episode_steps:
            terminated = True
            info["discount"] = 1.0  # Ensure infinite boostrap.
        info['is_success'] = bool(info['success'])
        terminated = terminated or info['is_success']
        self.current_obs = obs

        if self.use_reward_shaping:
            reward_info = {
                'reward': reward,
                'done': terminated or truncated,
                'info': info,
            }
            reward = self.reward_shaping(reward_info=reward_info)
            
        if self.use_camera:
            if self.do_shake:
                camera_config = copy.deepcopy(self.camera_config)
                camera_config.azimuth = self.shake_azimuth()
            else:
                camera_config = None
            self.render(camera_config=camera_config)
            current_fig_array = self.fig_arrays[-1][np.newaxis, ...]
            obs = self.tokenizer.get_latent_embedding(current_fig_array)
            
            if (terminated or truncated) and self.save_demo_path is not None:
                ### save the demo to the given path
                if info['is_success']:
                    save_demo_path = self.save_demo_path + '/succ'
                else:
                    save_demo_path = self.save_demo_path + '/fail'
                os.makedirs(name=save_demo_path, exist_ok=True)
                    
                fig_array = self.fig_arrays[-1]
                for _ in range(9):
                    self.fig_arrays.append(fig_array)
                fig_list = [Image.fromarray(item) for item in self.fig_arrays]

                version = 0
                # while os.path.exists(save_demo_path + f'/{self.env_name}_{version}.gif'):
                while os.path.exists(save_demo_path + f'/{self.env_name}_{round(reward, 4)}_{version}.gif'):
                    version += 1
                
                # imageio.mimsave(save_demo_path + f'/{self.env_name}_{version}.gif', fig_list, fps=20)
                imageio.mimsave(save_demo_path + f'/{self.env_name}_{round(reward, 4)}_{version}.gif', fig_list, fps=20)
                self.fig_arrays = self.fig_arrays[:-9]
            
            if (terminated or truncated) and self.collect_data:
                if len(self.fig_arrays) > self._max_episode_steps:
                    self.fig_arrays = self.fig_arrays[:self._max_episode_steps]
                fig_arrays = np.array(self.fig_arrays, dtype=np.uint8)
                info['fig_arrays'] = fig_arrays
                info['camera_id'] = self.camera_id
                info['camera_config'] = self.camera_config
                
        if self.is_gymnasium:
            return obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated or truncated, info

    def shake_azimuth(self):
        if self.shake_step == 0:
            self.shake_step = 10
            # self.shake_size = random.randint(-3, 3)
            self.shake_size = random.choice([-1, 1])
        
        self.current_azimuth += self.shake_size
        self.shake_step -= 1

        return self.current_azimuth
