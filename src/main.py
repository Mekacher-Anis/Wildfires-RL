import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from environment import ForestFireEnv, MDPConfig, ActionEnum
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import sys
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import importlib
import pygame
import os

from environment.env import get_action_name, print_observation
from environment.env_wrapper import CustomEnvWrapper

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


def train(cfg: BaseConfig) -> None:
    env: gym.Env = gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])
    module = importlib.import_module("stable_baselines3")
    AgentClass: BaseAlgorithm.__class__ = getattr(module, cfg["train"]["agent"])
    log_path = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        "tensorboard/",
    )
    model: BaseAlgorithm
    if cfg["trained_agent_path"]:
        print("Loading trained agent")
        model = AgentClass.load(cfg["trained_agent_path"], env=env)
    else:
        print("Training new agent")
        model = AgentClass(
            cfg["train"]["agent_policy"],
            env,
            verbose=cfg["verbose"],
            tensorboard_log=log_path,
        )
    cfg["MDP"]["eval_mode"] = True
    eval_env: gym.Env = gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])
    eval_env.reset()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=0,
    )
    model.learn(total_timesteps=cfg["train"]["total_timesteps"], callback=eval_callback)
    model.save(
        os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            "trained_agent.zip",
        ),
    )


def eval_trained_agent(cfg: BaseConfig) -> None:
    if not cfg["trained_agent_path"]:
        print("No trained agent path specified")
        return

    cfg["MDP"]["eval_mode"] = True
    env = gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])
    model = PPO.load(cfg["trained_agent_path"])

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


def record_trained_agent(cfg: BaseConfig) -> None:
    # will throw an error if not set
    sys.setrecursionlimit(100000)

    if not cfg["trained_agent_path"]:
        print("No trained agent path specified")
        return

    cfg["MDP"]["rendering"]["render_mode"] = "rgb_array"
    wrapped_env = CustomEnvWrapper([lambda: gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])])

    # Record the video starting at the first step
    env = VecVideoRecorder(
        wrapped_env,
        "videos",
        record_video_trigger=lambda x: x == 0,
        video_length=cfg["record"]["video_length"],
        name_prefix=f"trained-agent-ForestFireEnv-v0",
    )

    model = PPO.load(cfg["trained_agent_path"])

    n_steps = cfg["record"]["video_length"]

    obs = env.reset()
    sum_reward = 0
    for i in range(n_steps):
        for i in range(len(obs["map_state"])):
            print_observation(
                {
                    "map_state": obs["map_state"][i],
                    "resources": obs["resources"][i],
                }
            )
        action, _states = model.predict(obs)
        print(f"Action: {get_action_name(action[0])}")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}")
        sum_reward += reward
        if done:
            # when rendering the environment in rgb_array mode, the rendering is off 
            # by one step, so we need to manually trigger the last rendering here
            env.step([(0, 0, 0)])
            break
    print(f"Accumulated reward: {sum_reward}")
    env.close()


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
