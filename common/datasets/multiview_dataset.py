import numpy as np
import pickle
import random
import torch
import os
import copy
from torch.utils.data import Dataset, DataLoader

from common.utils import crop_sides_and_resize


class ImageDataLoader:
    def __init__(self, azimuths: list) -> None:
        self.data_dict = {}
        self.azimuths = [str(item) for item in azimuths]
        for azimuth in self.azimuths:
            self.data_dict[azimuth] = []
            
    def load_image(self, env_name, load_dir):
        with open(load_dir, 'rb') as f:
            trajs = pickle.load(f)
            for azimuth in self.azimuths:
                self.data_dict[azimuth] = self.data_dict[azimuth] + list(np.concatenate([traj['image'][azimuth] for traj in trajs], axis=0))
        for azimuth in self.azimuths:
            assert len(self.data_dict[azimuth]) == len(self.data_dict[self.azimuths[0]]) 

            
    def sample(self, batch_size, azimuth_num, identical_azimuth: bool = True):
        ### The sampled image batch data is: B * azimuth_num * W * H * 3
        assert azimuth_num <= len(self.azimuths)
        if identical_azimuth:
            sampled_azimuths = random.sample(self.azimuths, azimuth_num)
            sampled_image_ids = random.sample(list(range(len(self.data_dict[sampled_azimuths[0]]))), batch_size)
            sampled_images = []
            for azimuth in sampled_azimuths:
                sampled_azimuth_image = np.stack([self.data_dict[azimuth][id] for id in sampled_image_ids], axis=0)
                sampled_images.append(sampled_azimuth_image)
            batch_data = np.stack(sampled_images, axis=0).transpose(1, 0, 4, 2, 3)
        else:
            batch_data = []
            sampled_image_ids = random.sample(list(range(len(self.data_dict[self.azimuths[0]]))), batch_size)
            for id in sampled_image_ids:
               sampled_azimuths = random.sample(self.azimuths, azimuth_num)
               images = [self.data_dict[azimuth][id] for azimuth in sampled_azimuths]
               batch_data.append(np.stack(images, axis=0))
            batch_data = np.stack(batch_data, axis=0).transpose(0, 1, 4, 2, 3)  
        return batch_data
    
    def sample_with_nonequal_azimuth(self, ):
        batch_data = []
        sampled_image_ids = random.sample(list(range(len(self.data_dict[self.azimuths[0]]))), 8)
        sampled_azimuths = random.sample(self.azimuths, 8)
        for i, id in enumerate(sampled_image_ids):
            images = [self.data_dict[sampled_azimuths[i]][id]]
            batch_data.append(np.stack(images, axis=0))
        batch_data = np.stack(batch_data, axis=0).transpose(0, 1, 4, 2, 3)  
        return batch_data
    
    def sample_with_single_azimuth(self, batch_size, azimuth_id = 0):
        batch_data = []
        sampled_image_ids = random.sample(list(range(len(self.data_dict[self.azimuths[azimuth_id]]))), batch_size)
        for id in sampled_image_ids:
            sampled_azimuths = [self.azimuths[azimuth_id]]
            images = [self.data_dict[azimuth][id] for azimuth in sampled_azimuths]
            batch_data.append(np.stack(images, axis=0))
        batch_data = np.stack(batch_data, axis=0).transpose(0, 1, 4, 2, 3)  
        return batch_data
    
    
    def sample_with_azimuth_label(self, batch_size, azimuth_num):
        ### The sampled image batch data is: B * azimuth_num * W * H * 3
        assert azimuth_num <= len(self.azimuths)
        sampled_azimuth_ids = random.sample(list(range(len(self.azimuths))), azimuth_num)
        sampled_azimuths = [self.azimuths[id] for id in sampled_azimuth_ids]
        sampled_image_ids = random.sample(list(range(len(self.data_dict[sampled_azimuths[0]]))), batch_size)
        sampled_images = []
        for azimuth in sampled_azimuths:
            sampled_azimuth_image = np.stack([self.data_dict[azimuth][id] for id in sampled_image_ids], axis=0)
            sampled_images.append(sampled_azimuth_image)
        batch_data = np.stack(sampled_images, axis=0).transpose(1, 0, 4, 2, 3)
        azimuth_labels = np.array([sampled_azimuth_ids for _ in range(batch_size)], dtype=np.uint8)
        return batch_data, azimuth_labels
    
    
class ImageDataloaderWithPartialView:
    def __init__(self, 
                 train_envs: list, 
                 load_data_path: str,
                 camera_id_dict: dict, 
                 view_num: int=100, 
                 need_crop: bool=False
                 ) -> None:
        ##################
        # self.data_dict: a dict composed of the image data for each env, so the key for it is all the metaworld env name, the value for the dict is also a dict, of which the key is the camera id and the value is the images
        # self.env_name_dict: a dict whose key is the camera_id and the value is list of the env_name
        ##################
        self.train_data_dict = {}
        self.test_data_dict = {}
        self.camera_id_dict = camera_id_dict
        actual_view_num = 0
        for env_name in list(self.camera_id_dict.keys()):
            self.train_data_dict[env_name] = {}
            self.test_data_dict[env_name] = {}
            actual_view_num = max(actual_view_num, max(self.camera_id_dict[env_name] + [0]))
        
        try:    
            assert view_num == actual_view_num + 1
        except AssertionError as e:
            print(f"The actual view number is: {actual_view_num + 1}")
            view_num = actual_view_num + 1
        
        self.need_crop = need_crop
            
        self.env_name_dict = {}
        self.all_camera_ids = list(range(0, view_num))
        self.min_env_num = 100
        for i in range(view_num):
            self.env_name_dict[i] = []
            for env_name, camera_ids in self.camera_id_dict.items():
                if i in camera_ids:
                    self.env_name_dict[i].append(env_name)
            self.min_env_num = min(self.min_env_num, len(self.env_name_dict[i]))
            
        self.is_partial = False
        base_env_name = list(self.camera_id_dict.keys())[0]
        for env_name in list(self.camera_id_dict.keys()):
            if self.camera_id_dict[env_name] != self.camera_id_dict[base_env_name]:
                self.is_partial = True
                break

        # Load the image data
        for env_name in train_envs:
            self.load_image(env_name, load_dir=os.path.join(load_data_path, f"traj_{env_name}.pkl"))

            
    def load_image(self, env_name, load_dir):
        with open(load_dir, 'rb') as f:
            trajs = pickle.load(f)
            for i, camera_id in enumerate(self.camera_id_dict[env_name]):
                if self.need_crop:
                    all_images = list(np.concatenate([[crop_sides_and_resize(img) for img in list(traj['image'][camera_id])] for traj in trajs], axis=0))
                else:
                    all_images = list(np.concatenate([[img for img in list(traj['image'][camera_id])] for traj in trajs], axis=0))

                if i == 0:
                    all_image_ids = list(range(len(all_images)))
                    train_image_ids = random.sample(all_image_ids, int(0.9 * len(all_image_ids)))
                    test_image_ids = [id for id in all_image_ids if not id in train_image_ids]
                
                self.train_data_dict[env_name][camera_id] = [all_images[id] for id in train_image_ids]
                self.test_data_dict[env_name][camera_id] = [all_images[id] for id in test_image_ids]
        
        del trajs
                
    def sample_with_same_camera(self, batch_size, camera_num, image_per_env: int=1):
        #################
        # The sampled images are all with the same camera config
        # The final sampled image is with the shape: B * Camera_num * 3 * H * W
        #################
        assert batch_size // image_per_env <= self.min_env_num and batch_size % image_per_env == 0
        sampled_camera_ids = random.sample(self.all_camera_ids, camera_num)
        batch_data = []
        for i in range(camera_num):
            camera_id = sampled_camera_ids[i]
            sampled_envs = random.sample(self.env_name_dict[camera_id], batch_size // image_per_env)
            data = np.array([random.sample(self.train_data_dict[env_name][camera_id], image_per_env) for env_name in sampled_envs])
            batch_data.append(data)
            
        #### shape before transpose: Camera_num * (B // I) * I * w * h * 3
        batch_data = np.array(batch_data, dtype=np.uint8)
        CA, _, _, W, H, C = batch_data.shape
        assert C == 3
        batch_data = batch_data.transpose(1, 2, 0, 5, 3, 4)
        batch_data = batch_data.reshape((-1, CA, C, W, H))
        return batch_data
    
    def sample_with_same_env(self, batch_size, camera_num, image_per_env: int=1):
        #################
        # The sampled images are all with the same camera config
        # The final sampled image is with the shape: B * Camera_num * 3 * H * W
        #################
        assert batch_size % image_per_env == 0
        sampled_envs = random.sample(list(self.camera_id_dict.keys()), batch_size // image_per_env)
        batch_data = []
        for i in range(len(sampled_envs)):
            env_name = sampled_envs[i]
            image_ids = random.sample(list(range(len(list(self.train_data_dict[env_name].values())[0]))), image_per_env)
            for j in range(image_per_env):
                sampled_camera_ids = random.sample(self.camera_id_dict[env_name], camera_num)
                data = np.array([self.train_data_dict[env_name][camera_id][image_ids[j]] for camera_id in sampled_camera_ids])
                batch_data.append(data)
            
        #### shape before transpose: B * A * w * h * 3
        batch_data = np.array(batch_data, dtype=np.uint8)
        batch_data = batch_data.transpose(0, 1, 4, 2, 3)
        return batch_data
    
    def sample_for_eval(self, batch_size: int=3, camera_num: int=3):
        batch_data = [self.sample(batch_size=1, camera_num=camera_num).squeeze() for i in range(batch_size)]
        return np.array(batch_data, dtype=np.uint8)
    
    def sample_three_tuple(self, batch_size):
        sampled_envs = []
        sampled_camera_ids = []
        sampled_image_ids = []
        raw_data = []
        viewdiff_data = []
        latentdiff_data = []
        for i in range(batch_size):
            sampled_env = random.sample(list(self.camera_id_dict.keys()), 1)[0]
            sampled_camera_id = random.sample(list(self.camera_id_dict[sampled_env]), 1)[0]
            sampled_image_id = random.sample(list(range(len(self.train_data_dict[sampled_env][sampled_camera_id]))), 1)[0]    
            sampled_envs.append(sampled_env)
            sampled_camera_ids.append(sampled_camera_id)
            sampled_image_ids.append(sampled_image_id)
            raw_image = self.train_data_dict[sampled_env][sampled_camera_id][sampled_image_id]
            
            diff_camera_id = random.sample(list(self.camera_id_dict[sampled_env]), 1)[0]
            viewdiff_image = self.train_data_dict[sampled_env][diff_camera_id][sampled_image_id]
            
            diff_env = random.sample(list(self.env_name_dict[sampled_camera_id]), 1)[0]
            diff_image_id = random.sample(list(range(len(self.train_data_dict[diff_env][sampled_camera_id]))), 1)[0]
            latentdiff_image = self.train_data_dict[diff_env][sampled_camera_id][diff_image_id]
             
            raw_data.append(raw_image)
            viewdiff_data.append(viewdiff_image)
            latentdiff_data.append(latentdiff_image)
            
        raw_data = np.array(raw_data, dtype=np.uint8).transpose(0, 3, 1, 2)
        viewdiff_data = np.array(viewdiff_data, dtype=np.uint8).transpose(0, 3, 1, 2)
        latentdiff_data = np.array(latentdiff_data, dtype=np.uint8).transpose(0, 3, 1, 2)
            
        return raw_data, viewdiff_data, latentdiff_data
    
    def sample(self, batch_size, camera_num, seed=None, is_train=True):
        ### Here the sampled style is similar to the ImageDataloader Class
        if isinstance(seed, int):
            random.seed(seed)  # 设置 Python 的随机种子
            np.random.seed(seed)  # 设置 NumPy 的随机种子

        data_dict = self.train_data_dict if is_train else self.test_data_dict 
        
        if self.is_partial:
            while True:
                sampled_camera_ids = random.sample(list(self.env_name_dict.keys()), camera_num)
                valid_envs = []
                for env_name in list(self.camera_id_dict.keys()):
                    valid_flag = True
                    for camera_id in sampled_camera_ids:
                        if not camera_id in self.camera_id_dict[env_name]:
                            valid_flag = False
                            break
                    if valid_flag:
                        valid_envs.append(env_name)
                if len(valid_envs) >= 2:
                    break
        else:
            valid_envs = list(self.camera_id_dict.keys())
            sampled_camera_ids = random.sample(self.camera_id_dict[valid_envs[0]], camera_num)
        batch_data = []
        for i in range(batch_size):
            sampled_env = random.sample(valid_envs, 1)[0]
            sampled_image_id = random.sample(list(range(len(data_dict[sampled_env][sampled_camera_ids[0]]))), 1)[0]
            data = [data_dict[sampled_env][sampled_camera_id][sampled_image_id] for sampled_camera_id in sampled_camera_ids]
            batch_data.append(data)
            
        batch_data = np.array(batch_data, dtype=np.uint8).transpose(0, 1, 4, 2, 3)
        return batch_data
    
    def sample_latent(self, env_num, batch_size, camera_num, seed=None):
        ### Here the sampled style is similar to the ImageDataloader Class
        if isinstance(seed, int):
            random.seed(seed)  # 设置 Python 的随机种子
            np.random.seed(seed)  # 设置 NumPy 的随机种子 
        
        valid_envs = list(self.camera_id_dict.keys())
        sampled_camera_ids = random.sample(self.camera_id_dict[valid_envs[0]], camera_num)
        sampled_envs = random.sample(valid_envs, env_num)
        batch_data = []
        for sampled_env in sampled_envs:
            sub_batch_data = []
            sampled_span = random.randint(batch_size, 2 * batch_size)
            sampled_image_id_start = random.sample(list(range(len(self.train_data_dict[sampled_env][sampled_camera_ids[0]]) - sampled_span)), 1)[0]
            sampled_image_id_end = sampled_image_id_start + sampled_span + 1
            sampled_image_ids = random.sample(list(range(sampled_image_id_start, sampled_image_id_end)), batch_size)
            
            for sampled_image_id in sampled_image_ids:
                data = [self.train_data_dict[sampled_env][sampled_camera_id][sampled_image_id] for sampled_camera_id in sampled_camera_ids]
                sub_batch_data.append(data)
            batch_data.append(sub_batch_data)
            
        batch_data = np.array(batch_data, dtype=np.uint8).transpose(0, 1, 2, 5, 3, 4)
        return batch_data
    
    def sample_continuous(self, batch_size, camera_num, seed=None):
        ### Here the sampled style is similar to the ImageDataloader Class
        if isinstance(seed, int):
            random.seed(seed)  # 设置 Python 的随机种子
            np.random.seed(seed)  # 设置 NumPy 的随机种子 
        
        if self.is_partial:
            while True:
                sampled_camera_ids = random.sample(list(self.env_name_dict.keys()), camera_num)
                valid_envs = []
                for env_name in list(self.camera_id_dict.keys()):
                    valid_flag = True
                    for camera_id in sampled_camera_ids:
                        if not camera_id in self.camera_id_dict[env_name]:
                            valid_flag = False
                            break
                    if valid_flag:
                        valid_envs.append(env_name)
                if len(valid_envs) >= 2:
                    break
        else:
            valid_envs = list(self.camera_id_dict.keys())
            sampled_camera_ids = random.sample(self.camera_id_dict[valid_envs[0]], camera_num)
        batch_data = []
        sampled_env = random.sample(valid_envs, 1)[0]
        start = random.sample(list(range(len(self.train_data_dict[sampled_env][sampled_camera_ids[0]]) - batch_size)), 1)[0]
        for i in range(batch_size):
            sampled_image_id = start + i
            data = [self.train_data_dict[sampled_env][sampled_camera_id][sampled_image_id] for sampled_camera_id in sampled_camera_ids]
            batch_data.append(data)
            
        batch_data = np.array(batch_data, dtype=np.uint8).transpose(0, 1, 4, 2, 3)
        return batch_data
        
    
class ImageDatasetForPytorch(Dataset):
    def __init__(self, 
                 image_dataloader: ImageDataloaderWithPartialView, 
                 batch_size: int, 
                 camera_num: int,
                 seed: int, 
                 world_size: int = 100
                 ):
        
        random.seed(seed)  # 设置 Python 的随机种子
        np.random.seed(seed)  # 设置 NumPy 的随机种子
        torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
        
        self.image_dataloader = image_dataloader
        self.world_size = world_size
        self.batch_size = batch_size
        self.camera_num = camera_num
        
        self.data = [image_dataloader.sample(batch_size=batch_size, camera_num=camera_num) for _ in range(world_size)]
        
        self.idx_list = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.idx_list.append(idx)
        return_item = copy.deepcopy(self.data[idx])
        if len(self.idx_list) == len(self.data):
            self.idx_list = []
            self.data = [self.image_dataloader.sample(batch_size=self.batch_size, camera_num=self.camera_num) for _ in range(self.world_size)]                           
        return return_item
    
class ImageDataloaderForPyTorch(DataLoader):
    def __init__(self, 
                 dataset, 
                 sampler, 
                 batch_size=1, 
                 *args, 
                 **kwargs
                 ):
        super().__init__(dataset, 
                         batch_size=batch_size, 
                         sampler=sampler,
                         *args, 
                         **kwargs
                         )

    def __iter__(self):
        for batch in super().__iter__():
            yield self.process_batch(batch)

    def process_batch(self, batch):
        return batch

    
