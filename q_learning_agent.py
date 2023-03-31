import os, sys
import gymnasium as gym
import time
import numpy as np
import pickle

import text_flappy_bird_gym
from collections import defaultdict

class QLearningAgent:
    def agent_init(self, agent_init_info):
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])

        if agent_init_info["policy"]:
            self.q = agent_init_info["policy"]
        else:
            self.q = {}

        self.prev_action = None
        self.prev_state = None

    def agent_start(self, state):
        self.is_state_explored(state)
        #current_q = self.q[state]
        current_q = [self.q[state][0],  self.q[state][1]]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        self.is_state_explored(state)
        #current_q = self.q[state]
        current_q = [self.q[state][0],  self.q[state][1]]
        #print(current_q)

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
         # Perform an update if prev_state, prev_action, and reward are set
        #if self.prev_state is not None and self.prev_action is not None and reward is not None:
            #self.q[self.prev_state, self.prev_action] += self.step_size * (reward + self.discount * np.max(self.q[state]) - self.q[self.prev_state, self.prev_action])
        self.q[self.prev_state][self.prev_action] += self.step_size * (reward + self.discount * np.max(current_q) - self.q[self.prev_state][self.prev_action])

        self.prev_state = state
        self.prev_action = action
        #print(self.q)
        return action
    
    def agent_end(self, reward):
        #self.q[self.prev_state, self.prev_action] += self.step_size * (reward - self.q[self.prev_state, self.prev_action])
        self.q[self.prev_state][self.prev_action] += self.step_size * (reward - self.q[self.prev_state][self.prev_action])

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
    
    def save_agent(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(dict(self.q), f)

    def load_agent(self, file_path):
        with open(file_path, 'rb') as f:
            self.q = defaultdict(lambda: np.zeros(self.num_actions), pickle.load(f))
        self.prev_state = None
        self.prev_action = None
    
    # def is_state_explored(self, state):
    #   if state not in self.q.keys():
    #       self.q[state] = np.zeros(self.num_actions)

    def is_state_explored(self, state):
      if state not in self.q.keys():
          self.q[state] = {0: 0, 1: 0}

    def policy(self, state):
        #self.q[state]
        #current_q = self.q[state]
        current_q = [self.q[state][0],  self.q[state][1]]
        action = self.argmax(current_q)
        return action