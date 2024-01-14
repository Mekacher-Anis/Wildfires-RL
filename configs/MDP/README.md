# Environment configuration
The MDP configuration is loaded using hydra from [MDP.yaml](./MDP.yaml).\
Here's what the different values mean:
- `grid_size`: (default 100) how big is the map. If you change this, please make sure that you also set adequate values for the different resources list below; if the agent doesn't have enough resource and there are multiple fires set then it might not be able to put-out the fire.
- `forest_density`: (default 0.6) percentage of map covered by trees. If you set this to a small value, then the map will be sparse and the fire might not spread.
- `start_fires_num`: (default 1) how many trees on fire should the MDP start with. If you set this to a high value, let's take as an extreem the same number as the grid_size * grid_size (all blocks), then the agent won't have any change to set out the fire.
- `losing_reward`: (default -1000) reward/punishment value when the agent fails to put-out the fire, i.e. the whole forest burns.
- `tree_fire_spread_prob`: (default 1.0) probability of fire spreading from one tree to a direct neighbouring tree (up, down, right, left tree); if you set a small probability and the `forest_density` is too low the fire might go out by itself.
- `diagonal_tree_fire_spread_prob`: (default 0.57) probability of fire spreading from one tree to a diagonal neighbouring tree (up-right, bottom-left, etc); this is usually lower than `tree_fire_spread_prob` and it's ignored if it's higher (because diagonal trees are considered further away).
- `trench_fire_spread_prob`: (default 0.2) probability of fire spreading from one tree to a direct neighbouring **trench**.
- `diagonal_trench_fire_spread_prob`: (default 0.1) probability of fire spreading from one tree to a diagonal neighbouring **trench**.
- `disable_fire_propagation`: (default False) this is mostly used for testing, and it stops fire propagation. This is usefull if you set `start_fires_num` to high value and this to "True" to test what the different actions do.
- `budget`: (default 10000) the amount of money the agent starts with. Keep in mind that if this is set to a low value and the sum of the different actions multiplied by the number of actions available is higher than the budget, then the agent won't be able to use all the firefighters, firetrucks, etc...
- `firefighters`: (default 30) number of available firefighters to be used by the agent. Firefighters are used by the actions `ActionEnum.CONTROL_LINE`, and  `ActionEnum.BURNOUT`.
- `firetrucks`: (default 10) number of available firetrucks to be be used by the agent. Firetrucks are used by the action `ActionEnum.FIRETRUCK`.
- `helicopters`: (default 1) number of available firetrucks to be be used by the agent. Firetrucks are used by the action `ActionEnum.HELICOPTER`.
- `control_line_cost`: (default 30) cost of action `ActionEnum.CONTROL_LINE`, consider the available budget, and the other costs. The cost shouldn't only reflect "money" but also the environmental impact of the action (i.e. burning stuff or cutting tree is generally bad...).
- `burnout_cost`: (default 40) cost of action `ActionEnum.BURNOUT`, consider the available budget, and the other costs (same as above).
- `firetruck_cost`: (default 50) cost of action `ActionEnum.FIRETRUCK`, consider the available budget, and the other costs.
- `helicopter_cost`: (default 200) cost of action `ActionEnum.HELICOPTER`, consider the available budget, and the other costs, for example does it make sense to make the helicopter cheaper than the firetruck?
- `firetruck_success_rate`: (default 0.8) probability of the firetruck succeeding to put-out the fire in the blocks in its range (I made the default value up).
- `helicopter_success_rate`: (default 0.9) probability of the helicopter succeeding to put-out the fire in the blocks in its range (I made the default value up).
- `firetruck_range`: (default 7) radius of the circle in blocks from the center of action on which the firetruck action has effect. Please consider the `grid_size` when setting this value, a very large value might allow the agent to put out all the fires in the map with one action.
- `helicopter_range`: (default 20) radius of the circle in blocks from the center of action on which the helicopter action has effect. Please consider the `grid_size` when setting this value, a very large value might allow the agent to put out all the fires in the map with one action. Also consider the value `firetruck_range`, usually the helicopter range should be bigger but the helicopter should also cost more.
- `render_mode`: (default 'human') possible values are 'human' or 'rgb_array' or 'none'; please refer to [gym docs](https://gymnasium.farama.org/api/env/#gymnasium.Env.render)
- `render_fps`: (default 4) FPS of the rendered game.
- `window_size`: (default 500) size in pixel of the render window. Artificats might show up if this isn't divisible by `grid_size`.