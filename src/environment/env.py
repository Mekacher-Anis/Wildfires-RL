import gymnasium as gym
from omegaconf import DictConfig
import hydra
import numpy as np
import pygame
from enum import Enum
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


class ActionEnum(Enum):
    CONTROL_LINE = 0
    BURNOUT = 1
    FIRETRUCK = 2
    HELICOPTER = 3
    DO_NOTHING = 4


class StateEnum(Enum):
    EMPTY = 0
    TREE = 1
    FIRE = 2
    TRENCH = 3


def circle_indices(center: tuple[int, int], radius: int, matrix_shape: tuple[int, int]):
    """
    Get the indices of a circle around a center point.

    Parameters
    ----------
    center : tuple
        Center of the circle.
    radius : int
        Radius of the circle.
    matrix_shape : tuple
        Shape of the array.

    Returns
    ----------
    tuple
        Indices of the circle.
    """
    X, Y = np.ogrid[: matrix_shape[0], : matrix_shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return np.where(mask)


@dataclass
class EnvironmentConfig:
    grid_size: int = 100
    forest_density: float = 0.9
    start_fires_num: int = 1
    tree_fire_spread_prob: float = 1.0
    diagonal_tree_fire_spread_prob: float = 0.573
    trench_fire_spread_prob: float = 0.2
    diagonal_trench_fire_spread_prob: float = 0.1
    disable_fire_propagation: bool = False
    losing_reward: int = -1000
    disable_fire_propagation: bool = False


@dataclass
class ResourcesConfig:
    budget: int = 10000
    firefighters: int = 100
    firetrucks: int = 20
    helicopters: int = 1
    control_line_cost: int = 60
    burnout_cost: int = 60
    firetruck_cost: int = 50
    helicopter_cost: int = 500
    firetruck_range: int = 7
    helicopter_range: int = 15
    firetruck_success_rate: float = 0.8
    helicopter_success_rate: float = 0.9


@dataclass
class RenderingConfig:
    window_size: int = 500
    render_mode: str = "human"
    render_fps: int = 4


@dataclass
class MDPConfig:
    environment: EnvironmentConfig = EnvironmentConfig()
    resources: ResourcesConfig = ResourcesConfig()
    rendering: RenderingConfig = RenderingConfig()


cs = ConfigStore.instance()
cs.store(name="MDP", node=MDPConfig)


class ForestFireEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}

    def __init__(self, *, cfg: MDPConfig):
        super(ForestFireEnv, self).__init__()
        print("Initializing ForestFireEnv with following configuration:")
        self.grid_size = cfg["environment"].grid_size
        self.resources = cfg["resources"]
        self.environment = cfg["environment"]
        self.check_config()

        self.action_space = gym.spaces.MultiDiscrete(
            [
                len(ActionEnum),  # 6 action types
                self.grid_size,  # x coordinate
                self.grid_size,  # y coordinate
            ]
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(StateEnum),
            shape=(self.grid_size, self.grid_size),
            dtype=int,
        )

        # Initialize state
        self.state = self._init_state()

        self.window_size = cfg["rendering"]["window_size"]
        assert cfg["rendering"]["render_mode"] in ["human", "rgb_array", "none"]
        self.render_mode = cfg["rendering"]["render_mode"]
        self.render_fps = cfg["rendering"]["render_fps"]
        self.metadata["render_fps"] = self.render_fps

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window: pygame.Surface = None
        self.clock: pygame.time.Clock = None
        self.cell_size = self.window_size / self.grid_size

    def _init_state(self, seed: int = None):
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize the state to a grid of empty cells
        state = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Randomly select a number of cells to be trees
        num_trees = int(
            self.environment["forest_density"] * self.grid_size * self.grid_size
        )
        tree_indices = np.random.choice(
            self.grid_size * self.grid_size, num_trees, replace=False
        )
        state[
            np.unravel_index(tree_indices, (self.grid_size, self.grid_size))
        ] = StateEnum.TREE.value

        num_fires = self.environment["start_fires_num"]
        fire_indices = np.random.choice(tree_indices, num_fires, replace=False)
        state[
            np.unravel_index(fire_indices, (self.grid_size, self.grid_size))
        ] = StateEnum.FIRE.value

        return state

    def step(
        self, action: tuple[ActionEnum, int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Execute one time step within the environment
        # This is where you should implement the transition probabilities and rewards

        # Placeholder for state transition logic
        next_state = self.state.copy()
        reward = 0
        done = False

        # Unpack the action
        action_type, y, x = action

        action_applied = False
        # Apply the action
        if (
            action_type == ActionEnum.CONTROL_LINE.value
            and self.state[x, y] != StateEnum.FIRE.value
            and self.resources["firefighters"] > 0
        ):
            next_state[x, y] = StateEnum.TRENCH.value
            reward -= self.resources["control_line_cost"]
            self.resources["budget"] -= self.resources["control_line_cost"]
            self.resources["firefighters"] -= 1
            action_applied = True
        elif (
            action_type == ActionEnum.BURNOUT.value
            and self.state[x, y] == StateEnum.TREE.value
            and self.resources["firefighters"] > 0
        ):
            next_state[x, y] = StateEnum.EMPTY.value
            reward -= self.resources["burnout_cost"]
            self.resources["budget"] -= self.resources["burnout_cost"]
            self.resources["firefighters"] -= 1
            action_applied = True
        elif (
            action_type == ActionEnum.FIRETRUCK.value
            and self.resources["firetrucks"] > 0
        ):
            affected_blocks = circle_indices(
                (x, y), self.resources["firetruck_range"], next_state.shape
            )
            for i, j in zip(*affected_blocks):
                if (
                    (0 <= i < self.grid_size)
                    and (0 <= j < self.grid_size)
                    and next_state[i, j] == StateEnum.FIRE.value
                    and np.random.rand() < self.resources["firetruck_success_rate"]
                ):
                    next_state[i, j] = StateEnum.EMPTY.value
                    self.state[i, j] = StateEnum.EMPTY.value
            reward -= self.resources["firetruck_cost"]
            self.resources["budget"] -= self.resources["firetruck_cost"]
            self.resources["firetrucks"] -= 1
            action_applied = True
        elif (
            action_type == ActionEnum.HELICOPTER.value
            and self.resources["helicopters"] > 0
        ):
            affected_blocks = circle_indices(
                (x, y), self.resources["helicopter_range"], next_state.shape
            )
            for i, j in zip(*affected_blocks):
                if (
                    (0 <= i < self.grid_size)
                    and (0 <= j < self.grid_size)
                    and next_state[i, j] == StateEnum.FIRE.value
                    and np.random.rand() < self.resources["helicopter_success_rate"]
                ):
                    next_state[i, j] = StateEnum.EMPTY.value
                    self.state[i, j] = StateEnum.EMPTY.value
            reward -= self.resources["helicopter_cost"]
            self.resources["budget"] -= self.resources["helicopter_cost"]
            self.resources["helicopters"] -= 1
            action_applied = True

        # Update the fire spread
        if not self.environment["disable_fire_propagation"]:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.state[i, j] == StateEnum.FIRE.value:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                if (0 <= i + di < self.grid_size) and (
                                    0 <= j + dj < self.grid_size
                                ):
                                    next_state[
                                        i + di, j + dj
                                    ] = self.get_next_block_state(
                                        next_state[i + di, j + dj], abs(di) == abs(dj)
                                    )
                        next_state[i, j] = StateEnum.EMPTY.value

        # print(self.resources)

        # Check if done
        if StateEnum.FIRE.value not in next_state:
            print("Fire is out!!")
            done = True
            reward += (
                np.sum(next_state == StateEnum.TREE.value) + self.resources["budget"]
            )

        # no more budget
        if self.resources["budget"] <= 0:
            # print("Run out of money!!")
            done = True
            reward += self.environment["losing_reward"]

        # no more resources
        if (
            self.resources["firefighters"] <= 0
            and self.resources["firetrucks"] <= 0
            and self.resources["helicopters"] <= 0
        ):
            # print("Run out of resources!!")
            done = True
            reward += self.environment["losing_reward"]

        # Update state
        self.state = next_state

        if self.render_mode == "human":
            self._render_frame(action=action if action_applied else None)

        return next_state, reward, done, False, {}

    def get_next_block_state(self, current_state: StateEnum, diagonal: bool):
        """
        Given the current state of a block and the fact that there's fire a fire
        closeby, return the next state of the block based on the transition probabilities.
        """
        random_number = np.random.rand()
        if (
            current_state == StateEnum.TREE.value
            and random_number < self.environment["tree_fire_spread_prob"]
        ):
            if diagonal:
                return (
                    StateEnum.FIRE.value
                    if random_number
                    < self.environment["diagonal_tree_fire_spread_prob"]
                    else StateEnum.TREE.value
                )
            return StateEnum.FIRE.value
        elif (
            current_state == StateEnum.TRENCH.value
            and random_number < self.environment["trench_fire_spread_prob"]
        ):
            if diagonal:
                return (
                    StateEnum.FIRE.value
                    if random_number
                    < self.environment["diagonal_trench_fire_spread_prob"]
                    else StateEnum.TRENCH.value
                )
            return StateEnum.FIRE.value
        else:
            return current_state

    def reset(self, seed: int = None, options: dict = None):
        # Reset the state of the environment to an initial state
        self.state = self._init_state(seed=seed)
        if self.render_mode == "human":
            self._render_frame()
        return self.state, {}

    def render(self) -> None | np.ndarray:
        """Render frame"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(
        self, *, action: tuple[ActionEnum, int, int] = None
    ) -> None | np.ndarray:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # make window a bit taller to fit the resources
            # leave 150 pixels to the right for the buttons
            self.window = pygame.display.set_mode(
                (self.window_size + 150, self.window_size + 50)
            )
            self.window.fill((255, 255, 255))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # Render the environment using Pygame
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = (139, 69, 19)  # Dark brown color for empty cell
                if self.state[i, j] == StateEnum.TREE.value:
                    color = (0, 255, 0)  # Green for tree
                elif self.state[i, j] == StateEnum.FIRE.value:
                    color = (255, 0, 0)  # Red for fire
                elif self.state[i, j] == StateEnum.TRENCH.value:
                    color = (227, 145, 107)  # Light brown for trench
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        j * pix_square_size,
                        i * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                )

        # Render the action
        if action is not None:
            action_type, y, x = action
            color = (0, 0, 255)
            # action control line is already rendered
            if action_type == ActionEnum.BURNOUT.value:
                color = (255, 0, 255)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        y * pix_square_size,
                        x * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                )
            # for action firetruck and helicopter, render a circle in blue
            elif action_type == ActionEnum.FIRETRUCK.value:
                affected_blocks = circle_indices(
                    (x, y), self.resources["firetruck_range"], self.state.shape
                )
                for i, j in zip(*affected_blocks):
                    if (0 <= i < self.grid_size) and (0 <= j < self.grid_size):
                        pygame.draw.rect(
                            canvas,
                            (52, 171, 235),
                            pygame.Rect(
                                j * pix_square_size,
                                i * pix_square_size,
                                pix_square_size,
                                pix_square_size,
                            ),
                        )
            elif action_type == ActionEnum.HELICOPTER.value:
                affected_blocks = circle_indices(
                    (x, y), self.resources["helicopter_range"], self.state.shape
                )
                for i, j in zip(*affected_blocks):
                    if (0 <= i < self.grid_size) and (0 <= j < self.grid_size):
                        pygame.draw.rect(
                            canvas,
                            (52, 171, 235),
                            pygame.Rect(
                                j * pix_square_size,
                                i * pix_square_size,
                                pix_square_size,
                                pix_square_size,
                            ),
                        )

        if self.render_mode == "human" and self.window is not None:
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())

            # write resources on the button of the window over two lines
            font = pygame.font.SysFont("Arial", 20)
            text = font.render(
                f"Budget: {self.resources['budget']} | Firefighters: {self.resources['firefighters']} | Firetrucks: {self.resources['firetrucks']}       ",
                True,
                (0, 0, 0),
                (255, 255, 255),
            )
            self.window.blit(text, (0, self.window_size))
            text = font.render(
                f"Helicopters: {self.resources['helicopters']} | Fires left: {np.sum(self.state == StateEnum.FIRE.value)} | Trees left: {np.sum(self.state == StateEnum.TREE.value)}     ",
                True,
                (0, 0, 0),
                (255, 255, 255),
            )
            self.window.blit(text, (0, self.window_size + 25))

            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> bool:
        """Make sure environment is closed"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        return True

    def check_config(self):
        # verify that the configuration is valid
        assert (
            self.environment["tree_fire_spread_prob"] <= 1.0
        ), f"tree_fire_spread_prob is {self.environment['tree_fire_spread_prob']}"
        assert (
            self.environment["tree_fire_spread_prob"] >= 0.0
        ), f"tree_fire_spread_prob is {self.environment['tree_fire_spread_prob']}"
        assert (
            self.environment["trench_fire_spread_prob"] <= 1.0
        ), f"trench_fire_spread_prob is {self.environment['trench_fire_spread_prob']}"
        assert (
            self.environment["trench_fire_spread_prob"] >= 0.0
        ), f"trench_fire_spread_prob is {self.environment['trench_fire_spread_prob']}"
        assert (
            self.environment["diagonal_tree_fire_spread_prob"] <= 1.0
        ), f"diagonal_tree_fire_spread_prob is {self.environment['diagonal_tree_fire_spread_prob']}"
        assert (
            self.environment["diagonal_tree_fire_spread_prob"] >= 0.0
        ), f"diagonal_tree_fire_spread_prob is {self.environment['diagonal_tree_fire_spread_prob']}"
        assert (
            self.environment["diagonal_trench_fire_spread_prob"] <= 1.0
        ), f"diagonal_trench_fire_spread_prob is {self.environment['diagonal_trench_fire_spread_prob']}"
        assert (
            self.environment["diagonal_trench_fire_spread_prob"] >= 0.0
        ), f"diagonal_trench_fire_spread_prob is {self.environment['diagonal_trench_fire_spread_prob']}"
        assert (
            self.environment["tree_fire_spread_prob"]
            >= self.environment["diagonal_tree_fire_spread_prob"]
        ), f"tree_fire_spread_prob is {self.environment['tree_fire_spread_prob']} and diagonal_tree_fire_spread_prob is {self.environment['diagonal_tree_fire_spread_prob']}"
        assert (
            self.environment["trench_fire_spread_prob"]
            >= self.environment["diagonal_trench_fire_spread_prob"]
        ), f"trench_fire_spread_prob is {self.environment['trench_fire_spread_prob']} and diagonal_trench_fire_spread_prob is {self.environment['diagonal_trench_fire_spread_prob']}"
        assert (
            self.environment["forest_density"] <= 1.0
        ), f"forest_density is {self.environment['forest_density']}"
        assert (
            self.environment["forest_density"] >= 0.0
        ), f"forest_density is {self.environment['forest_density']}"
        assert (
            self.environment["start_fires_num"] <= self.grid_size * self.grid_size
        ), f"start_fires_num is {self.environment['start_fires_num']}"
        assert (
            self.environment["start_fires_num"] >= 0
        ), f"start_fires_num is {self.environment['start_fires_num']}"
        assert self.environment["disable_fire_propagation"] in [
            True,
            False,
        ], f"disable_fire_propagation is {self.environment['disable_fire_propagation']}"
        assert self.resources["budget"] >= 0, f"budget is {self.resources['budget']}"
        assert (
            self.resources["firefighters"] >= 0
        ), f"firefighters is {self.resources['firefighters']}"
        assert (
            self.resources["firetrucks"] >= 0
        ), f"firetrucks is {self.resources['firetrucks']}"
        assert (
            self.resources["helicopters"] >= 0
        ), f"helicopters is {self.resources['helicopters']}"
        assert (
            self.resources["control_line_cost"] >= 0
        ), f"control_line_cost is {self.resources['control_line_cost']}"
        assert (
            self.resources["burnout_cost"] >= 0
        ), f"burnout_cost is {self.resources['burnout_cost']}"
        assert (
            self.resources["firetruck_cost"] >= 0
        ), f"firetruck_cost is {self.resources['firetruck_cost']}"
        assert (
            self.resources["helicopter_cost"] >= 0
        ), f"helicopter_cost is {self.resources['helicopter_cost']}"
        assert (
            self.resources["firetruck_range"] >= 0
        ), f"firetruck_range is {self.resources['firetruck_range']}"
        assert (
            self.resources["helicopter_range"] >= 0
        ), f"helicopter_range is {self.resources['helicopter_range']}"
        assert (
            self.resources["firetruck_success_rate"] <= 1.0
        ), f"firetruck_success_rate is {self.resources['firetruck_success_rate']}"
        assert (
            self.resources["firetruck_success_rate"] >= 0.0
        ), f"firetruck_success_rate is {self.resources['firetruck_success_rate']}"
        assert (
            self.resources["helicopter_success_rate"] <= 1.0
        ), f"helicopter_success_rate is {self.resources['helicopter_success_rate']}"
        assert (
            self.resources["helicopter_success_rate"] >= 0.0
        ), f"helicopter_success_rate is {self.resources['helicopter_success_rate']}"

        # sum of all costs multiplied by number of available resources must be less than budget
        sum_costs = (
            max(self.resources["control_line_cost"], self.resources["burnout_cost"])
            * self.resources["firefighters"]
            + self.resources["firetruck_cost"] * self.resources["firetrucks"]
            + self.resources["helicopter_cost"] * self.resources["helicopters"]
        )
        assert (
            self.resources["budget"] >= sum_costs
        ), f"budget is {self.resources['budget']} and sum of all costs multiplied by number of available resources is {sum_costs}.\n Please refer to the documentation for more information about the configuration."

        # if the sum of costs is not equal to the budget, we need to warn the user
        if self.resources["budget"] != sum_costs:
            print(
                f"\n\nWARNING: budget is {self.resources['budget']} and sum of all costs multiplied by number of available resources is {sum_costs}.\n\n"
            )


gym.envs.register(id="ForestFireEnv-v0", entry_point=ForestFireEnv)
