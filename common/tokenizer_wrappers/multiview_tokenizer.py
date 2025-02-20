import torch
import numpy as np
import os
import copy
import warnings
import sys

project_dir = str(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_dir)
from common.models.multiview_vae import MultiViewBetaVAE

class MultiViewTokenizer:
    def __init__(self, config, device="cuda:0"):
        self.device = device
        self.model = MultiViewBetaVAE(config.view_encoder_config, config.latent_encoder_config, config.decoder_config, config.view_cb_config, config.latent_cb_config, config.img_size, config.patch_size, config.fusion_style, config.use_latent_vq, config.use_view_vq, config.is_view_ae).to(self.device)

        if os.path.exists(config.load_ckpt_path):
            self.model.load_checkpoint(config.load_ckpt_path)
            print(f"Load tokenizer from {config.load_ckpt_path} successfully!")
        else:
            warnings.warn(f"Unable to load tokenizer from {config.load_ckpt_path}, which means the tokenizer is randonly initialized!")

    def __call__(self, fig_array: np.ndarray):
        return self.get_latent_embedding(fig_array=fig_array)

    def get_latent_embedding(self, fig_array: np.ndarray):
        self.model.eval()
        if fig_array.shape[-1] == 3:
            fig_array = np.array(fig_array, dtype=np.uint8).transpose(0, 3, 1, 2)
        else:
            fig_array = np.array(fig_array, dtype=np.uint8)
        x = torch.tensor(fig_array / 127.5 - 1, dtype=torch.float).to(self.device)
        with torch.no_grad():
            _, _, z_l, _, _ = self.model.encode(x)

        z_l = z_l.reshape(z_l.shape[0], -1)
        return z_l.squeeze().cpu().numpy()
    
    def preprocess_image(self, trajs, batch_size: int = 64):
        self.model.eval()
        processed_trajs = []
        for traj in trajs:
            fig_array = traj["image"]
            fig_array = np.array(fig_array, dtype=np.uint8).transpose(0, 3, 1, 2) 
            image_embeddings = []  
            for id in range(0, fig_array.shape[0], batch_size):
                x = torch.tensor(fig_array[id:min(id+batch_size, fig_array.shape[0])] / 127.5 - 1, dtype=torch.float).to(self.device)
                with torch.no_grad():
                    _, _, z_l, _, _ = self.model.encode(x)
                z_l = z_l.reshape(z_l.shape[0], -1)
                image_embeddings.append(z_l)

            processed_embeddings = torch.cat(image_embeddings, dim=0)
            new_traj = copy.deepcopy(traj)
            del new_traj["image"]
            new_traj["obs"] = processed_embeddings.cpu().numpy()
            processed_trajs.append(new_traj)
        return processed_trajs
    

    
if __name__=='__main__':
    from common.tokenizer_wrappers import init_multiview_tokenizer
    tokenizer = init_multiview_tokenizer()
    print("Done.")