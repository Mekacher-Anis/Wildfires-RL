import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
import gymnasium as gym
import numpy as np


# Define a custom neural network architecture
class TwoHeads(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        print ("Obs space: ", obs_space)
        print ("Action space: ", action_space)
        print ("Num outputs: ", num_outputs)
        print ("Model config: ", model_config)
        print ("Name: ", name)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        print("Obsevartion space shape: ", int(np.product(obs_space.shape)))
        self.fc1 = nn.Linear(int(np.product(obs_space.shape)), 64)
        self.fc2 = nn.Linear(64, 64)
        print("Action space shape: ", action_space.shape)
        print("Num outputs: ", num_outputs)
        self.fc3 = nn.Linear(64, num_outputs)

    def forward(self, x, state=None, info=None):
        print("State: ", state)
        print("Info: ", info)
        print("X: ", x)
        raise "HOLY FOUCK"
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x