import gymnasium as gym
from typing import Optional
import numpy as np
from utils.evaluate import EvalMetricTracker
import copy

def collect_multi_view_episode(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
    must_success: bool=False
):
    success_label = False
    time = 0
    while not success_label:
        if init_obs is None or time > 0:
            obs = env.reset()
            time += 1
        else:
            obs = init_obs
            time += 1
        policy_obs = policy_env.reset()[0]

        # dataset.add(obs)
        episode_length = 0
        success_steps = 2
        is_terminal = False
        trajectory = {"image": [], "state": [], "action": [], "reward": [], "is_success": True}
        metric_tracker.reset()
        
        while not is_terminal:
            action = policy.get_action(policy_obs)
            if noise_type == "gaussian":
                action = action + epsilon * np.random.randn(*action.shape)
            elif noise_type == "uniform":
                action = action + epsilon * policy_env.action_space.sample()
            elif noise_type == "random":
                action = policy_env.action_space.sample()
            elif noise_type == 'no_noise':
                pass
            else:
                raise ValueError("Invalid noise type provided.")

            action = np.clip(action, -1 + 1e-5, 1 - 1e-5)  # Clip the action to the valid range after noise.
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy_obs, _, _, _, src_info = policy_env.step(action)
            metric_tracker.step(reward, info)
            is_terminal = terminated or truncated

            episode_length += 1

            # If the other env finishes we have to terminate
            if getattr(policy_env, "curr_path_length", 0) == policy_env.max_path_length:
                is_terminal = True  # set done to true manually for meta world.
                info["fig_arrays"] = env.unwrapped.get_arrays()

            if info["success"]:
                success_steps -= 1
            
            if success_steps == 0:
                is_terminal = True
                
            if episode_length == 500:
                is_terminal = True
            
            if next_obs.max() > 2:
                is_terminal = True
            
            trajectory['state'].append(obs)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            if is_terminal:
                trajectory['image'] = copy.deepcopy(info['fig_arrays'])
                trajectory['is_success'] = bool(info['is_success'])
            obs = next_obs
        if must_success:
            success_label = trajectory['is_success']
        else:
            success_label = True
            
    return trajectory

def collect_single_view_episode(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
    must_success: bool=False
):
    success_label = False
    time = 0
    while not success_label:
        if init_obs is None or time > 0:
            obs = env.reset()
            time += 1
        else:
            obs = init_obs
            time += 1
        policy_obs = policy_env.reset()[0]

        # dataset.add(obs)
        episode_length = 0
        success_steps = 2
        is_terminal = False
        trajectory = {"image": [], "state": [], "action": [], "reward": [], "is_success": True}
        metric_tracker.reset()
        
        while not is_terminal:
            action = policy.get_action(policy_obs)
            if noise_type == "gaussian":
                action = action + epsilon * np.random.randn(*action.shape)
            elif noise_type == "uniform":
                action = action + epsilon * policy_env.action_space.sample()
            elif noise_type == "random":
                action = policy_env.action_space.sample()
            elif noise_type == 'no_noise':
                pass
            else:
                raise ValueError("Invalid noise type provided.")

            action = np.clip(action, -1 + 1e-5, 1 - 1e-5)  # Clip the action to the valid range after noise.
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy_obs, _, _, _, src_info = policy_env.step(action)
            metric_tracker.step(reward, info)
            is_terminal = terminated or truncated

            episode_length += 1

            # If the other env finishes we have to terminate
            if getattr(policy_env, "curr_path_length", 0) == policy_env.max_path_length:
                is_terminal = True  # set done to true manually for meta world.
                info["fig_arrays"] = env.unwrapped.get_arrays()

            if info["success"]:
                success_steps -= 1
            
            if success_steps == 0:
                is_terminal = True
                
            if episode_length == 500:
                is_terminal = True
            
            if next_obs.max() > 2:
                is_terminal = True
            
            trajectory['state'].append(obs)
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            if is_terminal:
                trajectory['image'] = copy.deepcopy(info['fig_arrays'])
                trajectory['is_success'] = bool(info['is_success'])
                trajectory['camera_id'] = env.unwrapped.camera_id
                trajectory['camera_config'] = env.unwrapped.camera_config
            obs = next_obs
        if must_success:
            success_label = trajectory['is_success']
        else:
            success_label = True
            
    return trajectory
