import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict, Discrete, Space
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.misc import SlimFC, normc_initializer


class ConvActionMaskModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        self.layer1 = nn.Sequential(
            self._init_weights(nn.Conv2d(1, 8, kernel_size=3, stride=3, padding=2), 0.01), # 4x4x8
            nn.ReLU()
        ) # 4x4x8
        self.layer2 = nn.Sequential(
           self._init_weights(nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0), 0.01), # 2x2x16
           nn.ReLU()
        ) # 2x2x16
        #self.drop_out = nn.Dropout()
        self.fc1 = self._init_weights(nn.Linear(2 * 2 * 16 + 4, 50), 0.01)
        self.fc2 = self._init_weights(nn.Linear(50, num_outputs), 0.01)
        self._value_branch = SlimFC(
            in_size=50,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        self._features = None
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)
#        self.to(torch.device("cuda"))

    def _init_weights(self, layer: nn.Module, std: float = 1.0) -> nn.Module:
        tensor = layer.weight
        tensor.data.normal_(0, 1)
        tensor.data *= std / torch.sqrt(tensor.data.pow(2).sum(1, keepdim=True))
        return layer

    def forward(self, input_dict, state, seq_lens):
        inputs = input_dict["obs"]["actual_obs"]["map_state"]
        inputs = inputs[:, None, :, :]
        inputs = inputs
        print("layer1 device", next(self.layer1.parameters()).device)
        print("input device", inputs.device)
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
#        out = self.drop_out(out)
        out = torch.cat((out, input_dict["obs"]["actual_obs"]["resources"]), dim=1)
        self._features = self.fc1(out)
        logits = self.fc2(self._features)
        if self.no_masking:
            return logits, state
        action_mask = torch.cat(input_dict["obs"]["action_mask"], dim=1)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        mask_expanded = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
        mask_expanded[:, : inf_mask.shape[1]] = inf_mask
        masked_logits = logits + mask_expanded
        return masked_logits, state

    def value_function(self):
        return self._value_branch(self._features).squeeze(1)
