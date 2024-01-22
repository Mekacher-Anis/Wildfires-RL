import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict, Discrete, Space
from ray.rllib.utils.torch_utils import FLOAT_MIN


class ActionMaskModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "actual_obs" in orig_space.spaces
        )
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        self.attention_mask_size = int(
            np.sum([p.shape[0] for p in orig_space["action_mask"]])
        )
        self.internal_model = FullyConnectedNetwork(
            Box(
                -1.0,
                1.0,
                (obs_space.shape[0] - self.attention_mask_size,),
                obs_space.dtype,
            ),
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # action_mask = input_dict["obs"]["action_mask"][0]
        fc_inputs = input_dict["obs_flat"][..., : -self.attention_mask_size]
        logits, _ = self.internal_model(
            {"obs": fc_inputs},
            state,
            seq_lens,
        )
        if self.no_masking:
            return logits, state
        action_mask = torch.cat(input_dict["obs"]["action_mask"], dim=1)
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        # mask_expanded = torch.zeros(logits.shape)
        # mask_expanded[:, : action_mask.shape[1]] = inf_mask
        # masked_logits = logits + mask_expanded
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()
