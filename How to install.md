# How to install

## Installing Gym
Follow the instructions on [Gym GitHub Page](https://github.com/openai/gym)

## Installing Mujoco
Reference: [Mujoco GitHub Page](https://github.com/openai/mujoco-py)

1. cd ~/.mujoco
2. git clone https://github.com/openai/mujoco-py.git
3. cd mujoco-py
4. pip install -r requirements.txt
5. pip install -r requirements.dev.txt
6. python setup.py install

## Installing the Rocket Lander environment
1. Copy the file `rocket_lander.py` to the folder `$HOME/anaconda3/envs/gym/lib/python3.5/site-packages/gym/envs/box2d`

2. Add `from gym.envs.box2d.rocket_lander import RocketLander` to the file `__init__.py` located in this same folder

3. Add the following code to the file '__init__.py' in the folder `$HOME/anaconda3/envs/gym/lib/python3.5/site-packages/gym/envs`

```
register(
    id='RocketLander-v0',
    entry_point='gym.envs.box2d:RocketLander',
    max_episode_steps=1000,
    reward_threshold=0,
)
```

## Correcting error with Box2D

1. `conda install -c https://conda.anaconda.org/kne pybox2d`
