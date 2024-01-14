# Fighting wildfires with Reinforcement Learning

This project offers a [Gymnasium](https://gymnasium.farama.org/index.html) environment that simulates a wildfire and an agent that can do various actions in order to put-out the fire.

## Setup
1. Create vitual python environment
  ```shell
  conda create -n wildfires python=3.9
  ```
2. Activate environment
  ```shell
  conda activate wildfires
  ```
3. Install dependencies
  ```shell
  pip install -r requirements.txt
  ```
4. Play the game!!
  ```shell
    python3 src/main.py +action=play +MDP=MDP_basic
  ```
PS: Change the MDP configuration in [environment](./configs/environment/README.md) and see what happens 😉
## Train
Train a PPO agent:
```
python3 src/main.py +action=train +MDP=MDP_basic +train=train_PPO_single_run_basic
```
OR
add a custom configuration under [configs/train](./configs/train/) and call
```shell
python3 src/main.py +action=train +MDP=MDP_basic +train=custom_config
```
Don't forget to launch Tensorboard to see your logs ;)
```shell
tensorboard --logdir='./logs'
```
OR when using vscode run command "Python: Launch Tensorboard"
## Why?
Accoring to different resources [[1](https://www.wri.org/insights/global-trends-forest-fires)] [[2](https://sgp.fas.org/crs/misc/IF10244.pdf)] millions of acres of green forests are lost every year due to wildfires and this leads to a vicious cycle of more wildfires due to the carbon emissions from the fire as shown in the follow picture:
![fire-climate-feedback-loop-wri.png](https://files.wri.org/d8/s3fs-public/styles/965_wide/s3/2023-08/fire-climate-feedback-loop-wri.png?VersionId=uGo_Op7ZGn.lHtdd_4GL32pGfQHWht7W&itok=8Zv1AymP)
This also leads to what is called "Extreme wildfires" [[3](https://www.sciencedirect.com/science/article/abs/pii/B9780128157213000011)] that are beyond our human capacity and current technology to put them out / limit them.
This is where the use of Reinforcement Learning would help, what if the AI can more efficiently put-out wildfires?

## MDP
Our MDP will be built on the ["Forest-fire Model"](https://www.wikiwand.com/en/Forest-fire_model) which is based on 4 simple rules:
1. A burning cell turns into an empty cell
2. A tree will burn if at least one neighbor is burning
3. A tree ignites with probability f even if no neighbor is burning
4. An empty space fills with a tree with probability p

Some ideas in our MDP are also similar to [this paper](https://doi.org/10.1109/IJCNN48605.2020.9207548)
Because in our task we're more cocerned about putting out the fires than what happens after the fires (i.e. trees regrowing) rule number 4 will be dropped. This also has the nice benefit of making the MDP episodic; i.e the game ends when all trees burn-out or the fire is put-out.
Another point regarding rule number 3, the environment will start with a random set of trees on fire (how many is hyperparameter) and no trees will self-ignite dynamically after the start.
The agent in our case will act as the Fire Department / Goverment and has certain resources at its disposal which it can use to fight the fires.

### State
- Grid world with X x Y cells (this can also be a hyperparameter to test how different world shapes affect the agent strategy)
- Cells have four possible states: Empty/Earth, Tree, Fire, Trench (dug up from firefighters)
- Agent has $a$ number of firefighters
- Agent has $b$ number of firetrucks
- Agent has $c$ number of [helicopter / planes](https://www.wikiwand.com/en/Aerial_firefighting)
- Agent has a budget of size $d$ (money)

### Actions
- Send firefighters to a specific location and do action [[6](https://www.mentalfloss.com/article/57094/10-strategies-fighting-wildfires)]
	- Control line: stops fire from spreading in a certain direction a long a virtual wall (e.g. trench) with probability $P_{a1}$ and costs $C_{a1}$
	- Burnout: removes trees along a one dimensional line with a max length to stop fire from spreading; removing the trees will work with 100% probability and will cost $C_{a2}$
- Send firetruck to a specific location to put-out fire with probability $P_{a4}$ and costs $C_{a4}$
- Send helicopters / planes to a specific location to put-out fire with probability $P_{a5}$ and costs $C_{a5}$

### Transition probabilities
- Fire will spread in case of no action taken at the given location as described by the Forest-fire model
- All actions that have a cost will reduce the agents budget
- Action "control line" will replace the cells along a certain line by "trench" cells
- Action "burnout" will replace tree cells a long a line with empty cells
- Action "firetruck" will put out fires that exist in the given location with probability  $P_{a4}$
- Action "helicopter/plane" will put out fires at the given location with proabability $P_{a5}$
- Termination
	- Agent uses all resources it has
	- Fire is put out
	- Fire cosumes the whole map

### Rewards
- Fire stopped: reward is number of trees still standing + budget remaining
- Agent does action: negative rewards, i.e. associated costs with that action
- Fire consumes entire map: complete and utter failure, -10000 reward
## Hyperparameters
- All the transition probabilites, costs, and actions listed above
- Size and shape of the environment
- Probability of fire spreading; higher values allows us to simulate casees of. "extreme wildfires"
## Further improvements
- Simulate wind
- Simulate wildfires with different hotspots

### References
- [1] https://www.wri.org/insights/global-trends-forest-fires
- [2] https://sgp.fas.org/crs/misc/IF10244.pdf
- [3] https://www.sciencedirect.com/science/article/abs/pii/B9780128157213000011
- [4] https://www.wikiwand.com/en/Forest-fire_model
- [5] https://www.wikiwand.com/en/Aerial_firefighting
- [6] https://www.mentalfloss.com/article/57094/10-strategies-fighting-wildfires
- [7] https://doi.org/10.1109/IJCNN48605.2020.9207548