import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from stable_baselines3 import A2C, PPO, SAC
from environment import ForestFireEnv, MDPConfig, ActionEnum
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import sys
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import importlib
import pygame

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
    trained_agent_path: str = "trained_agent"
    video_length: int = 1000


cs.store(name="record", node=RecordConfig)


@dataclass
class BaseConfig:
    env: str = "MDP_basic"
    verbose: int = 1
    tensorboard_log: str = "./logs/"
    train: TrainConfig = TrainConfig()
    record: RecordConfig = RecordConfig()
    MDP: MDPConfig = MDPConfig()


cs.store(name="base", node=BaseConfig)


def train(cfg: BaseConfig) -> None:
    env: gym.Env = gym.make("ForestFireEnv-v0", cfg=cfg["MDP"])
    module = importlib.import_module("stable_baselines3")
    AgentClass = getattr(module, cfg["train"]["agent"])
    model = AgentClass(
        cfg["train"]["agent_policy"],
        env,
        verbose=cfg["verbose"],
        tensorboard_log=cfg["tensorboard_log"],
    )
    model.learn(total_timesteps=cfg["train"]["total_timesteps"])
    model.save("trained_agent.zip")


def record_trained_agent(cfg: BaseConfig) -> None:
    # will throw an error if not set
    sys.setrecursionlimit(100000)

    env = DummyVecEnv(
        [lambda: gym.make("ForestFireEnv-v0", render_mode="rgb_array", cfg=cfg["MDP"])]
    )

    # Record the video starting at the first step
    env = VecVideoRecorder(
        env,
        "videos",
        record_video_trigger=lambda x: x == 0,
        video_length=cfg["record"]["video_length"],
        name_prefix=f"trained-agent-ForestFireEnv-v0",
    )

    model = PPO.load(cfg["trained_agent_path"])

    n_steps = cfg["record"]["video_length"]

    obs = env.reset()
    for i in range(n_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break
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
    else:
        print("Unknown action")
        return


if __name__ == "__main__":
    main()
