import gymnasium as gym
from omegaconf import DictConfig
import hydra
import numpy as np
import pygame
from enum import Enum
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from copy import deepcopy


class ActionEnum(Enum):
    DO_NOTHING = 0
    CONTROL_LINE = 1
    BURNOUT = 2
    FIRETRUCK = 3
    HELICOPTER = 4


def get_action_name(action: tuple[ActionEnum, int, int]):
    action_type, y, x = action
    # action_type = int(action_type * len(ActionEnum) - 1)
    # y, x = int(y * (4 - 1)), int(x * (4 - 1))
    if action_type == ActionEnum.CONTROL_LINE.value:
        return f"Control line at ({x}, {y})"
    elif action_type == ActionEnum.BURNOUT.value:
        return f"Burnout at ({x}, {y})"
    elif action_type == ActionEnum.FIRETRUCK.value:
        return f"Firetruck at ({x}, {y})"
    elif action_type == ActionEnum.HELICOPTER.value:
        return f"Helicopter at ({x}, {y})"
    elif action_type == ActionEnum.DO_NOTHING.value:
        return "Do nothing"
    else:
        return "Unknown action"


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
    control_line_cost: int = 60
    burnout_cost: int = 60
    firetruck_cost: int = 50
    helicopter_cost: int = 500
    firetruck_range: int = 7
    helicopter_range: int = 15
    firetruck_success_rate: float = 0.8
    helicopter_success_rate: float = 0.9


@dataclass
class ResourcesConfig:
    budget: int = 10000
    firefighters: int = 100
    firetrucks: int = 20
    helicopters: int = 1


@dataclass
class RenderingConfig:
    window_size: int = 500
    render_mode: str = "human"
    render_fps: int = 4


@dataclass
class MDPConfig:
    eval_mode: bool = False
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
        self.cfg = cfg
        self.grid_size = cfg["environment"].grid_size
        self.environment = cfg["environment"].copy()

        self.action_space = gym.spaces.MultiDiscrete(
            [
                len(ActionEnum),  # 6 action types
                self.grid_size,  # x coordinate
                self.grid_size,  # y coordinate
            ]
        )

        self.observation_space = gym.spaces.Dict(
            {
                "map_state": gym.spaces.Box(
                    low=0,
                    high=len(StateEnum),
                    shape=(self.grid_size, self.grid_size),
                    dtype=int,
                ),
                # not dict because nested dict is not supported by stable-baselines3
                "resources": gym.spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(4,),  # budget, firefighters, firetrucks, helicopters
                    dtype=int,
                ),
            }
        )

        self.affected_blocks: np.ndarray[bool] = None
        """ A boolean array that indicates which blocks were affected by agent's actions. """
        self.state = self._init_state()
        self.check_config()

        self.window_size = cfg["rendering"]["window_size"]
        assert cfg["rendering"]["render_mode"] in ["human", "rgb_array", "none"]
        self.render_mode = cfg["rendering"]["render_mode"]
        self.render_fps = cfg["rendering"]["render_fps"]
        self.metadata["render_fps"] = self.render_fps
        self.eval_mode = cfg["eval_mode"] if "eval_mode" in cfg else False
        """ Is the current environment being used for evaluation? """
        self.last_action: tuple[ActionEnum, int, int] = None
        """ Used to render the action taken by the agent in rgb_array mode. """

        self.window: pygame.Surface = None
        """ If human-rendering is used, `self.window` will be a reference to the window that we draw to. """
        self.clock: pygame.time.Clock = None
        """
        `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.cell_size = self.window_size / self.grid_size
        """ How many pixels on the window correspond to a single cell in the grid. """

    def _init_state(self, seed: int = None):
        # set seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize the state to a grid of empty cells
        map_state = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Randomly select a number of cells to be trees
        num_trees = int(
            self.environment["forest_density"] * self.grid_size * self.grid_size
        )
        tree_indices = np.random.choice(
            self.grid_size * self.grid_size, num_trees, replace=False
        )
        map_state[
            np.unravel_index(tree_indices, (self.grid_size, self.grid_size))
        ] = StateEnum.TREE.value

        num_fires = self.environment["start_fires_num"]
        fire_indices = np.random.choice(tree_indices, num_fires, replace=False)
        map_state[
            np.unravel_index(fire_indices, (self.grid_size, self.grid_size))
        ] = StateEnum.FIRE.value
        self.affected_blocks = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.last_action = None

        return {
            "map_state": map_state,
            "resources": np.array(
                [
                    self.cfg["resources"]["budget"],
                    self.cfg["resources"]["firefighters"],
                    self.cfg["resources"]["firetrucks"],
                    self.cfg["resources"]["helicopters"],
                ]
            ),
        }

    def step(
        self, action: tuple[ActionEnum, int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        next_state = deepcopy(self.state)
        reward = 0
        done = False
        self.last_action = None
        """
        how many fires were put out directly by the chosen action
        or by previous actions that affected fire propagation
        """
        num_of_put_out_fires = 0

        action_type, y, x = action
        num_old_fires = np.sum(self.state["map_state"] == StateEnum.FIRE.value)

        if self.eval_mode:
            print(self.state["resources"])

        action_applied = False
        if (
            action_type == ActionEnum.CONTROL_LINE.value
            and self.state["map_state"][x, y] != StateEnum.FIRE.value
            and self.state["resources"][1] > 0
        ):
            next_state["map_state"][x, y] = StateEnum.TRENCH.value
            next_state["resources"][0] -= self.environment["control_line_cost"]
            next_state["resources"][1] -= 1
            self.affected_blocks[x, y] = True
            action_applied = True
        elif (
            action_type == ActionEnum.BURNOUT.value
            and self.state["map_state"][x, y] == StateEnum.TREE.value
            and self.state["resources"][1] > 0
        ):
            next_state["map_state"][x, y] = StateEnum.EMPTY.value
            next_state["resources"][0] -= self.environment["burnout_cost"]
            next_state["resources"][1] -= 1
            self.affected_blocks[x, y] = True
            action_applied = True
        elif (
            action_type == ActionEnum.FIRETRUCK.value and self.state["resources"][2] > 0
        ):
            affected_blocks = circle_indices(
                (x, y),
                self.environment["firetruck_range"],
                next_state["map_state"].shape,
            )
            for i, j in zip(*affected_blocks):
                if (
                    (0 <= i < self.grid_size)
                    and (0 <= j < self.grid_size)
                    and next_state["map_state"][i, j] == StateEnum.FIRE.value
                    and np.random.rand() < self.environment["firetruck_success_rate"]
                ):
                    next_state["map_state"][i, j] = StateEnum.EMPTY.value
                    self.state["map_state"][i, j] = StateEnum.EMPTY.value
                    num_of_put_out_fires += 1
            next_state["resources"][0] -= self.environment["firetruck_cost"]
            next_state["resources"][2] -= 1
            action_applied = True
        elif (
            action_type == ActionEnum.HELICOPTER.value
            and self.state["resources"][3] > 0
        ):
            affected_blocks = circle_indices(
                (x, y),
                self.environment["helicopter_range"],
                next_state["map_state"].shape,
            )
            for i, j in zip(*affected_blocks):
                if (
                    (0 <= i < self.grid_size)
                    and (0 <= j < self.grid_size)
                    and next_state["map_state"][i, j] == StateEnum.FIRE.value
                    and np.random.rand() < self.environment["helicopter_success_rate"]
                ):
                    next_state["map_state"][i, j] = StateEnum.EMPTY.value
                    self.state["map_state"][i, j] = StateEnum.EMPTY.value
                    num_of_put_out_fires += 1
            next_state["resources"][0] -= self.environment["helicopter_cost"]
            next_state["resources"][3] -= 1
            action_applied = True

        # Update the fire spread
        if not self.environment["disable_fire_propagation"]:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.state["map_state"][i, j] == StateEnum.FIRE.value:
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                if (0 <= i + di < self.grid_size) and (
                                    0 <= j + dj < self.grid_size
                                ):
                                    next_state["map_state"][
                                        i + di, j + dj
                                    ] = self.get_next_block_state(
                                        next_state["map_state"][i + di, j + dj],
                                        abs(di) == abs(dj),
                                    )
                                    # if the fire didn't spread we assume it's the agent's actions that put it out
                                    if (
                                        next_state["map_state"][i + di, j + dj]
                                        != StateEnum.FIRE.value
                                        and self.affected_blocks[i + di, j + dj]
                                    ):
                                        num_of_put_out_fires += 1
                        next_state["map_state"][i, j] = StateEnum.EMPTY.value

        # Check if done
        if StateEnum.FIRE.value not in next_state["map_state"]:
            if self.eval_mode:
                print("Fire is out!!")
            done = True
            # if there are less than 10% trees left, we lose
            # we can't test for 0 because the fire might not have spread to all trees
            if (
                np.sum(next_state["map_state"] == StateEnum.TREE.value)
                < 0.1 * self.grid_size * self.grid_size * self.environment["forest_density"]
            ):
                reward += self.environment["losing_reward"]
            else:
                reward += np.sum(next_state["map_state"] == StateEnum.TREE.value)

        # no more budget
        if self.state["resources"][0] <= 0:
            if self.eval_mode:
                print("Run out of money!!")
            done = True
            reward += self.environment["losing_reward"]

        # no more resources
        if (
            self.state["resources"][1] <= 0
            and self.state["resources"][2] <= 0
            and self.state["resources"][3] <= 0
        ):
            if self.eval_mode:
                print("Run out of resources!!")
            done = True
            reward += self.environment["losing_reward"]

        # negative reward for fire spreading
        fires_diff = num_old_fires - np.sum(
            next_state["map_state"] == StateEnum.FIRE.value
        )
        # reward += fires_diff * 10

        # reward for putting out fires
        # reward += num_of_put_out_fires * 100

        # negative reward for doing a useless action
        # if reward == 0 and not action_applied:
        #     reward += -100

        # Update state
        self.state = next_state

        if action_applied:
            self.last_action = action

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self._render_frame(
                action=action if action_applied else None,
                reward=reward,
            )

        if self.eval_mode:
            print(f"Action: {action}, Reward: {reward}, Done: {done}")

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
        self, *, action: tuple[ActionEnum, int, int] = None, reward: float = None
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
                if self.state["map_state"][i, j] == StateEnum.TREE.value:
                    color = (0, 255, 0)  # Green for tree
                elif self.state["map_state"][i, j] == StateEnum.FIRE.value:
                    color = (255, 0, 0)  # Red for fire
                elif self.state["map_state"][i, j] == StateEnum.TRENCH.value:
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
        action = action if action is not None else self.last_action
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
                    (x, y),
                    self.environment["firetruck_range"],
                    self.state["map_state"].shape,
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
                    (x, y),
                    self.environment["helicopter_range"],
                    self.state["map_state"].shape,
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
                f"Budget: {self.state['resources'][0]} | Firefighters: {self.state['resources'][1]} | Firetrucks: {self.state['resources'][2]}       ",
                True,
                (0, 0, 0),
                (255, 255, 255),
            )
            self.window.blit(text, (0, self.window_size))
            text = font.render(
                f"Helicopters: {self.state['resources'][3]} | Fires left: {np.sum(self.state['map_state'] == StateEnum.FIRE.value)} | Trees left: {np.sum(self.state['map_state'] == StateEnum.TREE.value)}  |  Reward: {reward}   ",
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
        assert (
            self.state["resources"][0] >= 0
        ), f"budget is {self.state['resources'][0]}"
        assert (
            self.state["resources"][1] >= 0
        ), f"firefighters is {self.state['resources'][1]}"
        assert (
            self.state["resources"][2] >= 0
        ), f"firetrucks is {self.state['resources'][2]}"
        assert (
            self.state["resources"][3] >= 0
        ), f"helicopters is {self.state['resources'][3]}"
        assert (
            self.environment["control_line_cost"] >= 0
        ), f"control_line_cost is {self.environment['control_line_cost']}"
        assert (
            self.environment["burnout_cost"] >= 0
        ), f"burnout_cost is {self.environment['burnout_cost']}"
        assert (
            self.environment["firetruck_cost"] >= 0
        ), f"firetruck_cost is {self.environment['firetruck_cost']}"
        assert (
            self.environment["helicopter_cost"] >= 0
        ), f"helicopter_cost is {self.environment['helicopter_cost']}"
        assert (
            self.environment["firetruck_range"] >= 0
        ), f"firetruck_range is {self.environment['firetruck_range']}"
        assert (
            self.environment["helicopter_range"] >= 0
        ), f"helicopter_range is {self.environment['helicopter_range']}"
        assert (
            self.environment["firetruck_success_rate"] <= 1.0
        ), f"firetruck_success_rate is {self.environment['firetruck_success_rate']}"
        assert (
            self.environment["firetruck_success_rate"] >= 0.0
        ), f"firetruck_success_rate is {self.environment['firetruck_success_rate']}"
        assert (
            self.environment["helicopter_success_rate"] <= 1.0
        ), f"helicopter_success_rate is {self.environment['helicopter_success_rate']}"
        assert (
            self.environment["helicopter_success_rate"] >= 0.0
        ), f"helicopter_success_rate is {self.environment['helicopter_success_rate']}"

        # sum of all costs multiplied by number of available resources must be less than budget
        sum_costs = (
            max(
                self.environment["control_line_cost"],
                self.environment["burnout_cost"],
            )
            * self.state["resources"][1]
            + self.environment["firetruck_cost"] * self.state["resources"][2]
            + self.environment["helicopter_cost"] * self.state["resources"][3]
        )
        assert (
            self.state["resources"][0] >= sum_costs
        ), f"budget is {self.state['resources'][0]} and sum of all costs multiplied by number of available resources is {sum_costs}.\n Please refer to the documentation for more information about the configuration."

        # if the sum of costs is not equal to the budget, we need to warn the user
        if self.state["resources"][0] != sum_costs:
            print(
                f"\n\nWARNING: budget is {self.state['resources'][0]} and sum of all costs multiplied by number of available resources is {sum_costs}.\n\n"
            )


gym.envs.register(id="ForestFireEnv-v0", entry_point=ForestFireEnv)
