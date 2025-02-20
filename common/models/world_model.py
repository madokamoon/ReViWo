from .modules.wm_transformer import WMTransformer
import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, wm_config):
        super().__init__()
        self.world_model = WMTransformer(wm_config)

    def forward(self, x, a):
        return self.world_model(x, a)

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))