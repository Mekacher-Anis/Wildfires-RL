import gymnasium as gym
import hydra
from omegaconf import OmegaConf

# from stable_baselines3 import A2C, PPO, SAC
from environment import ForestFireEnv, MDPConfig, ActionEnum
from stable_baselines3.common.vec_env import VecVideoRecorder
import sys
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import pygame
from environment.env import get_action_name, print_observation
from environment.env_wrapper import CustomEnvWrapper
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from ray import tune
import ray
from ray import tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from models import TwoHeads
from models.action_mask_model import ActionMaskModel
from PIL import Image
from datetime import datetime
import os
import cv2

cs = ConfigStore.instance()


@dataclass
class TrainConfig:
    agent: str = "PPO"  # A2C, PPO, SAC
    agent_policy: str = "MlpPolicy"
    total_timesteps: int = 1000000


cs.store(name="train", node=TrainConfig)


@dataclass
class RecordConfig:
    render_fps: int = 4
    video_length: int = 1000


cs.store(name="record", node=RecordConfig)


@dataclass
class BaseConfig:
    env: str = "MDP_basic"
    verbose: int = 1
    trained_agent_path: str = ""
    train: TrainConfig = TrainConfig()
    record: RecordConfig = RecordConfig()
    MDP: MDPConfig = MDPConfig()


cs.store(name="base", node=BaseConfig)


def unpack_config(env):
    class UnpackedEnv(env):
        def __init__(self, env_config):
            super().__init__(**env_config)

    return UnpackedEnv


def train(cfg: BaseConfig) -> None:
    # Initialize Ray
    ray.init()

    eval_cfg = cfg.copy()
    eval_cfg["MDP"]["eval_mode"] = True

    # Define the training configuration
    config: AlgorithmConfig = {
        "env": unpack_config(ForestFireEnv),
        "env_config": {"cfg": cfg["MDP"]},
        "evaluation_config": {
            "env": unpack_config(ForestFireEnv),
            "env_config": {"cfg": eval_cfg["MDP"]},
        },
        "evaluation_interval": 2,
        "num_gpus": 0,
        "num_workers": 8,
        "train_batch_size": 40000,
        "model": {
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no_masking": False,
            },
            "fcnet_hiddens": [256, 256],
        },
    }

    # Define the directory for logging
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Start the training
    tune.run(
        PPO,
        config=config,
        local_dir=log_dir,
        checkpoint_config={
            "checkpoint_frequency": 1,
            "num_to_keep": 3,
            "checkpoint_at_end": True,
        },
        stop={
            "training_iteration": cfg["train"]["total_timesteps"]
        },  # Define when to stop training
    )


def eval_trained_agent(cfg: BaseConfig) -> None:
    if not cfg["trained_agent_path"]:
        print("No trained agent path specified")
        return

    # Initialize Ray
    ray.init()

    eval_cfg = cfg.copy()
    eval_cfg["MDP"]["eval_mode"] = True

    # Define the training configuration
    config: AlgorithmConfig = {
        "env": unpack_config(ForestFireEnv),
        "env_config": {"cfg": cfg["MDP"]},
        "evaluation_config": {
            "env": unpack_config(ForestFireEnv),
            "env_config": {"cfg": eval_cfg["MDP"]},
        },
        "evaluation_interval": 1,
        "evaluation_duration": 10000,
        "num_gpus": 0,
        "num_workers": 1,
        "create_env_on_driver=True": True,
        "model": {
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no_masking": False,
            },
            "fcnet_hiddens": [256, 256],
        },
    }

    # Create a new trainer and restore from the checkpoint
    trainer = PPO(config=config)
    trainer.restore(cfg["trained_agent_path"])
    eval_res = trainer.evaluate()
    print("Episode reward mean: ", eval_res["evaluation"]["episode_reward_mean"])
    print("Episode length mean: ", eval_res["evaluation"]["episode_len_mean"])


def record_trained_agent(cfg: BaseConfig) -> None:
    # will throw an error if not set
    sys.setrecursionlimit(100000)

    if not cfg["trained_agent_path"]:
        print("No trained agent path specified")
        return

    cfg["MDP"]["rendering"]["render_mode"] = "rgb_array"
    env: ForestFireEnv = gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])

    # Initialize Ray
    ray.init()

    eval_cfg = cfg.copy()
    eval_cfg["MDP"]["eval_mode"] = True

    # Define the training configuration
    config: AlgorithmConfig = {
        "env": unpack_config(ForestFireEnv),
        "env_config": {"cfg": cfg["MDP"]},
        "evaluation_config": {
            "env": unpack_config(ForestFireEnv),
            "env_config": {"cfg": eval_cfg["MDP"]},
        },
        "evaluation_interval": 1,
        "num_gpus": 0,
        "num_workers": 1,
        "create_env_on_driver=True": True,
        "train_batch_size": 4000,
        "model": {
            "custom_model": ActionMaskModel,
            "custom_model_config": {
                "no_masking": False,
            },
            "fcnet_hiddens": [256, 256],
        },
    }

    # Create a new trainer and restore from the checkpoint
    trainer = PPO(config=config)
    trainer.restore(cfg["trained_agent_path"])

    n_steps = cfg["record"]["video_length"]

    obs, _ = env.reset()
    output_folder_name = datetime.now().strftime("./videos/recording_%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_folder_name)
    images_paths: list[str] = []
    Image.fromarray(env.render()).save(f"{output_folder_name}/frame_0.png")
    images_paths.append(f"{output_folder_name}/frame_0.png")
    sum_reward = 0
    for i in range(n_steps):
        # for i in range(len(obs["map_state"])):
        print_observation(obs)
        action = trainer.compute_single_action(obs)
        print(f"Action: {get_action_name(action)}")
        obs, reward, done, trunc, info = env.step(action)
        Image.fromarray(env.render()).save(f"{output_folder_name}/frame_{i + 1}.png")
        images_paths.append(f"{output_folder_name}/frame_{i + 1}.png")
        print(f"Reward: {reward}")
        sum_reward += reward
        if done:
            # when rendering the environment in rgb_array mode, the rendering is off
            # by one step, so we need to manually trigger the last rendering here
            env.step((0, 0, 0))
            Image.fromarray(env.render()).save(
                f"{output_folder_name}/frame_{i + 2}.png"
            )
            images_paths.append(f"{output_folder_name}/frame_{i + 2}.png")
            break
    print(f"Accumulated reward: {sum_reward}")
    env.close()

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        f"{output_folder_name}/output.mp4",
        fourcc,
        cfg["MDP"]["rendering"]["render_fps"],
        (
            cfg["MDP"]["rendering"]["window_size"],
            cfg["MDP"]["rendering"]["window_size"],
        ),
    )

    for file in images_paths:
        frame = cv2.imread(file)
        print(frame.shape)
        video.write(frame)

    # Release the VideoWriter and close OpenCV windows
    video.release()
    cv2.destroyAllWindows()


def play(cfg: BaseConfig) -> None:
    cfg["MDP"]["rendering"]["render_mode"] = "human"
    env: ForestFireEnv = gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])
    env.reset()

    # Pygame setup
    BUTTON_WIDTH, BUTTON_HEIGHT = 120, 40
    FONT = pygame.font.Font(None, 20)
    MAP_SIZE = cfg["MDP"]["rendering"]["window_size"]
    BLACK = (0, 0, 0)

    # Create a list of buttons
    buttons: list[tuple[pygame.Rect, ActionEnum]] = []
    for i, action in enumerate(ActionEnum):
        rect = pygame.Rect(
            MAP_SIZE + 10,
            15 + (BUTTON_HEIGHT + 10) * i,
            BUTTON_WIDTH,
            BUTTON_HEIGHT,
        )
        buttons.append((rect, action))

    # Draw the buttons
    for button in buttons:
        pygame.draw.rect(env.unwrapped.window, BLACK, button[0], 2)
        label_surface = FONT.render(button[1].name, True, BLACK)
        env.unwrapped.window.blit(
            label_surface,
            button[0].center - pygame.Vector2(label_surface.get_size()) / 2,
        )

    # Main game loop
    running = True
    current_action = ActionEnum.DO_NOTHING
    mouse_down = False
    while running:
        action = (ActionEnum.DO_NOTHING, 0, 0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for button in buttons:
                    if button[0].collidepoint(event.pos):
                        current_action = button[1].value
                clickx, clicky = event.pos
                if clickx < MAP_SIZE and clicky < MAP_SIZE:
                    action = (
                        current_action,
                        int(clickx // env.unwrapped.cell_size),
                        int(clicky // env.unwrapped.cell_size),
                    )
                    mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
            elif event.type == pygame.MOUSEMOTION and mouse_down:
                clickx, clicky = event.pos
                if clickx < MAP_SIZE and clicky < MAP_SIZE:
                    action = (
                        current_action,
                        int(clickx // env.unwrapped.cell_size),
                        int(clicky // env.unwrapped.cell_size),
                    )
        ns, r, done, trunc, info = env.step(action)
        running = not done

    pygame.quit()
    sys.exit()


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: BaseConfig) -> None:
    if "action" not in cfg:
        print("No action specified")
        return

    print(OmegaConf.to_yaml(cfg))
    if cfg["action"] == "train":
        train(cfg)
    elif cfg["action"] == "record":
        record_trained_agent(cfg)
    elif cfg["action"] == "play":
        play(cfg)
    elif cfg["action"] == "eval":
        eval_trained_agent(cfg)
    else:
        print("Unknown action")
        return


if __name__ == "__main__":
    main()
