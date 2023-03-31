import os, sys
import gymnasium as gym
import time
import numpy as np
import pickle

import text_flappy_bird_gym
from q_learning_agent import QLearningAgent  # Import the QLearningAgent class from your agent file

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    obs, info = env.reset()

    # Load
    agent_file_path = "trained_agent.pkl"
    with open(agent_file_path, 'rb') as f:
        loaded_policy = pickle.load(f)

    # Create and load the trained agent
    agent_info = {"num_actions": 2, "epsilon": 0.1, "step_size": 0.1, "discount": 0.99, "seed": 42, "policy": loaded_policy}
    agent = QLearningAgent()
    agent.agent_init(agent_info)
    #agent_file_path = "trained_agent.pkl"
    #agent.load_agent(agent_file_path)

    # iterate
    while True:

        # Select next action using the trained agent
        action = agent.policy(state=obs)  # Use the trained agent to choose the action

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2)  # FPS

        # If player is dead break
        if done:
            break

    env.close()
