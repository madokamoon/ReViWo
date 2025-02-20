import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from offlinerlkit.nets import EnsembleLinear


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()
        self.activation_o = nn.Hardtanh(min_val=-0.2, max_val=1.2)
        # self.activation_o = nn.Hardtanh(min_val=-1.2, max_val=1.2)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        if self.obs_dim > 256:
            self.obs_dim_low = 128
            obs_dim = self.obs_dim_low
            module_list_obs = []
            hidden_dims_obs = [self.obs_dim, 2048, 512, obs_dim]
            weight_decays_obs = [2.5e-5 for _ in range(len(hidden_dims_obs)-1)]
            for in_dim, out_dim, weight_decay in zip(hidden_dims_obs[:-1], hidden_dims_obs[1:], weight_decays_obs):
                module_list_obs.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
            self.obs_layers_in = nn.ModuleList(module_list_obs)
            self.action_layer_in = EnsembleLinear(self.action_dim, self.action_dim, num_ensemble, 2.5e-5)

            module_list_obs = []
            hidden_dims_obs.reverse()
            for in_dim, out_dim, weight_decay in zip(hidden_dims_obs[:-1], hidden_dims_obs[1:], weight_decays_obs):
                module_list_obs.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
            self.obs_layers_out = nn.ModuleList(module_list_obs)

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(self.obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(self.obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)

        if hasattr(self, 'obs_layers_in'):
            obs = obs_action[..., :self.obs_dim]
            action = obs_action[..., self.obs_dim:]
            assert action.shape[-1] == self.action_dim
            for layer in self.obs_layers_in:
                obs = self.activation(layer(obs))
            action = self.action_layer_in(action)
            obs_action = torch.cat([obs, action], dim=-1)
            
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        output = self.output_layer(output)
        # output = self.activation_o(output)
        mean, logvar = torch.chunk(output, 2, dim=-1)

        if hasattr(self, 'obs_layers_out'):
            next_obs = mean[..., :self.obs_dim_low]
            reward = mean[..., self.obs_dim_low:]
            assert reward.shape[-1] == 1
            for layer in self.obs_layers_out:
                next_obs = self.activation(layer(next_obs))
            mean = torch.cat([next_obs, reward], dim=-1)

            next_obs = logvar[..., :self.obs_dim_low]
            reward = logvar[..., self.obs_dim_low:]
            assert reward.shape[-1] == 1
            for layer in self.obs_layers_out:
                next_obs = self.activation(layer(next_obs))
            logvar = torch.cat([next_obs, reward], dim=-1)

        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs