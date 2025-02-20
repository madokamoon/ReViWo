import torch
import torch.nn.functional as F
import torch.nn as nn
import json, argparse
from PIL import Image
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
project_dir = str(os.path.dirname(os.path.dirname(__file__)))

try:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
except:
    pass


ALL_ENVIRONMENTS = [
    "assembly-v2", 
    "basketball-v2", 
    "button-press-v2", 
    "door-open-v2", 
    "window-close-v2", 
    "drawer-open-v2", 
    "dial-turn-v2", 
    "soccer-v2", 
    "handle-pull-side-v2", 
    "reach-v2"]
ALL_ENVIRONMENTS += [
    'reach-v2',
    'push-v2',
    'pick-place-v2',
    'button-press-v2',
    'door-unlock-v2',
    'door-open-v2',
    'window-open-v2',
    'faucet-open-v2',
    'coffee-push-v2',
    'coffee-button-v2',      
]
ALL_ENVIRONMENTS = sorted(list(set(ALL_ENVIRONMENTS)))


def sobel_edge_detection(img):
    # 定义Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 将Sobel算子转为适用于输入图像的形式
    sobel_x = sobel_x.to(img.device)
    sobel_y = sobel_y.to(img.device)
    
    # 对输入图像应用Sobel算子
    img = img.unsqueeze(1)  # 增加通道维度
    edge_x = F.conv2d(img, sobel_x, padding=1)
    edge_y = F.conv2d(img, sobel_y, padding=1)
    
    # 计算边缘图
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    return edge.squeeze(1)


def create_weight_map(img, edge_threshold=0.1, high_weight=5.0, low_weight=1.0):
    # 使用Sobel算子检测边缘
    edges = sobel_edge_detection(img)
    
    # 根据边缘强度生成权重图
    weight_map = torch.where(edges > edge_threshold, high_weight, low_weight)
    return weight_map.view(img.size(0), -1)


def create_adaptive_weight_map(input_img, reconstructed_img, high_weight=10.0, low_weight=1.0, threshold2= 10 / 255):
    # 计算差异图
    difference = torch.abs(input_img - reconstructed_img)
    
    # 根据差异生成权重图
    weight_map = torch.where(difference >= threshold2, high_weight, low_weight)
    return weight_map


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight_map):
        loss = (input - target) ** 2
        weighted_loss = loss * weight_map
        return torch.mean(weighted_loss)
    
def normalize_tensor(input: torch.Tensor, normalize_together: bool = False):
    if normalize_together:
        max = input.abs().max().detach()
        return input / max
    else:
        normalized_tensor = torch.zeros_like(input).to(input.device)
        for i in range(input.shape[0]):
            distance = torch.sum(input[i] ** 2)
            normalized_tensor[i, :] = input[i] / distance
        return normalized_tensor 
            

def get_lr_lambda(initial_lr, final_lr, num_epochs):
    def lr_lambda(epoch):
        if epoch <= num_epochs:
            return (final_lr / initial_lr) ** (epoch / num_epochs)
        else:
            return final_lr / initial_lr
    return lr_lambda

def set_train_test_env(train_env_idxs, test_env_idxs):
    train_envs = list(ALL_V2_ENVIRONMENTS.keys())[train_env_idxs[0]:train_env_idxs[1]]
    test_envs = list(ALL_V2_ENVIRONMENTS.keys())[test_env_idxs[0]:test_env_idxs[1]]
    return train_envs, test_envs

def save_args_to_file(args_list, args_name_list, filename):
    save_dict = {}
    for i in range(len(args_list)):
        save_dict[args_name_list[i]] = vars(args_list[i])
    with open(filename, 'w') as f:
        json.dump(save_dict, f)
        
def load_args_from_file(args_name_list, filename):
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    res = []
    for args_name in args_name_list:
        res.append(argparse.Namespace(**args_dict[args_name]))
    return res


def crop_sides_and_resize(image: np.ndarray, crop_percentage = 0.1):
        assert image.shape == (128, 128, 3)
        img = Image.fromarray(image)
        
        # 获取原始尺寸
        original_width, original_height = img.size
        
        # 计算裁剪尺寸
        crop_amount = int(original_width * crop_percentage)
        new_width = original_width - (2 * crop_amount)  # 左右各裁剪 crop_amount
        
        # 计算裁剪的起始点
        left = crop_amount
        top = crop_amount
        right = original_width - crop_amount
        bottom = original_width - crop_amount
        
        # 裁剪图片
        cropped_img = img.crop((left, top, right, bottom))
        
        # 将裁剪后的图片放大到原始宽度
        resized_img = cropped_img.resize((original_width, original_height), Image.LANCZOS)
        
        return np.array(resized_img, dtype=np.uint8)


def compute_similarity(a, b, dim, way: str = "cosine-similarity", lower_bound: float = 0.0):
        simi = 0
        if way == "cosine-similarity":
            simi = F.cosine_similarity(a, b, dim=dim)
        elif way == "l2-distance":
            simi = -((a - b) ** 2).sum(dim=dim)
        elif way == "mixed":
            simi = 0.1 * F.cosine_similarity(a, b, dim=dim) - ((a - b) ** 2).sum(dim=dim)
        else:
            raise NotImplementedError
        return simi - lower_bound

def visualize_indices(encoding_indices, save_dir):
    plt.figure()
    plt.hist(list(encoding_indices), bins=16)
    plt.title('Histogram of the encoding tokens')
    plt.xlabel('Token id')
    plt.ylabel('Frequency')
    plt.savefig(save_dir + "/token frequency.jpg")
    plt.close()
    
    image = Image.open(save_dir + "/token frequency.jpg")
    return np.array(image, dtype=np.uint8)

def load_camera_id_config(load_path, train_envs):
    with open(os.path.join(load_path, "camera_configs.pkl"), "rb") as f:
        camera_ids, camera_configs = pickle.load(f)
            
    camera_id_dict = {}
    camera_config_dict = {}
    for env_name in train_envs:
        camera_id_dict[env_name] = camera_ids
        camera_config_dict[env_name] = camera_configs

    return camera_id_dict, camera_config_dict