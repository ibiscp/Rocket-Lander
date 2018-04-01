import numpy as np
import gym
from gym import wrappers as wr
import time

# Define if using memoryTree or not
memoryTree = False
if memoryTree:
    from agentTree import Agent
else:
    from agent import Agent

# Agent
gamma = 0.99                        # reward discount factor
learning_rate = 0.00025             # learning rate
num_episodes = 10000                # number of episodes to train
update_target = 10000  #1000        # steps to update target model
epsilon_start = 1.0                 # start epsilon
epsilon_end = 0.01                  # minimum epsilon
epsilon_decay = 0.00001             # speed of decay
save_model_episode = 100            # interval to save model

# Brain
batch_size = 64 #128                # size of batch from experience replay memory for updates

# Memory
memory_capacity = 100000 # 100000   # capacity of experience replay memory

# Environment
environment = 'LunarLander-v2' #'RocketLander-v0'     # Environment name

# folders
monitorDir = 'videos'
modelDir = 'models'

# Start environment
env = gym.make(environment)

# State and action variables
stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

# set seeds to 0
env.seed(0)
np.random.seed(0)

# Initialize agent
agent = Agent(stateCnt, actionCnt, memory_capacity, update_target, batch_size,
epsilon_start, epsilon_end, epsilon_decay, learning_rate, environment, gamma)

# Load model if exists
agent.loadModel(modelDir)

# Populate memory
while agent.memory.numberSamples() < memory_capacity:
    print('Loading memory: %7i/%7i'%(agent.memory.numberSamples(),memory_capacity))
    state = env.reset()
    done = False

    while not done:
        # act
        action = agent.act(state)

        # execute action
        next_state, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1, 1)

        # observe
        if memoryTree:
            agent.memory.add(abs(reward), (state, action, reward, None if done else next_state))
        else:
            agent.memory.add((state, action, reward, None if done else next_state))

        # update next sate
        state = next_state

print('\nMemory Loaded: %7i/%7i\n'%(agent.memory.numberSamples(),memory_capacity))

# Reset environment episode
#env = wr.Monitor(env, monitorDir, resume=True, video_callable=lambda episode_id: episode_id%100==0 or episode_id==1, uid=agent.uid)
env.episode_id = agent.episode

import os

# Train the model
for ep in range(agent.episode, num_episodes + 1):
    total_reward = 0
    steps_in_episode = 0

    try:
        state = env.reset()
    except:
        agent.saveModel(modelDir)
    done = False

    #env.render()

    while not done:
        # act
        action = agent.act(state)

        # execute action
        next_state, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1, 1)

        # update total reward
        total_reward += reward

        #env.render()

        # observe
        agent.observe((state, action, reward, None if done else next_state))

        # decrement epsilon
        agent.decrementEpsilon()

        # replay
        agent.replay()

        # update variables
        state = next_state
        steps_in_episode += 1

    # append data to history
    agent.appendReward(ep, steps_in_episode, total_reward)

    # save model
    if (ep%save_model_episode==0):
        agent.saveModel(modelDir)

    print('Episode %4i, Reward: %8.3f, Steps: %4i, Next eps: %6.4f, Total steps: %7i'%(ep,total_reward,steps_in_episode, agent.epsilon, agent.steps))

print('\n\nLearning finished!\n\nPlaying games!\n')

# Set epsilon to 0
agent.epsilon = 0
ep = 0

while True:
    ep += 1
    total_reward = 0
    steps_in_episode = 0

    # reset and render
    state = env.reset()
    env.render()
    done = False

    while not done:
        # act
        action = agent.act(state)

        # execute action
        next_state, reward, done, _ = env.step(action)
        reward = np.clip(reward, -1, 1)

        # render
        env.render()

        # update total reward
        total_reward += reward

        # update variables
        state = next_state
        steps_in_episode += 1

    print('Episode %4i, Reward: %8.3f, Steps: %4i'%(ep,total_reward,steps_in_episode))
