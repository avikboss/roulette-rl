# -*- coding: utf-8 -*-
"""ReinforcementLearningProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k1SoI7HMKVhhc86ts5qWBt5-qIsuOX8H
"""

import gym
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from collections import defaultdict

def sarsa(episodes=2000, alpha=0.01, gamma=0.9, eps=0.1):
    env = gym.make("Roulette-v0")
    env.reset()
    q = defaultdict(lambda: defaultdict(int))
    rmse_list = []
    policy_list = []

    for i in range(episodes):
        s = env.observation_space.sample()
        policy = defaultdict(int)
        a = get_eps_greedy_action(env, eps, policy)
        done = False

        while not done:
            next_s, reward, done, _ = env.step(a)
            next_a = get_eps_greedy_action(env, eps, policy)
            q[s][a] = q[s][a] + alpha * (reward + gamma*q[next_s][next_a] - q[s][a])
            policy[s] = max(q[s], key=q[s].get)
            s = next_s
            a = next_a
        rmse_list.append(rmse(q,env))
        policy_list.append(policy[0])
        env.reset()

    policy = {s:max(q[s], key=q[s].get) for s in range(env.observation_space.n)}
    return q, policy, rmse_list, policy_list

def qlearning(episodes=2000, alpha=0.01, gamma=0.9, eps=0.1):

    env = gym.make("Roulette-v0")
    env.reset()
    q = defaultdict(lambda: defaultdict(int))
    rmse_list = []
    policy_list = []

    for _ in range(episodes):

        s = env.observation_space.sample()
        policy = defaultdict(int)

        done = False
        while not done:
            a = get_eps_greedy_action(env,eps, policy)
            next_s, reward, done, _ = env.step(a)
            q[s][a] = q[s][a] + alpha * (reward + gamma*q[next_s][max(q[next_s],key=q[next_s].get)] - q[s][a])
            policy[s] = max(q[s], key=q[s].get)

            s = next_s

        rmse_list.append(rmse(q,env))
        policy_list.append(policy[0])
        env.reset()

    policy = {s:max(q[s], key=q[s].get) for s in range(env.observation_space.n)}
    return q, policy, rmse_list, policy_list

def get_eps_greedy_action(env,eps,policy):
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return policy[0]

def rmse(observed_q, env):
    residuals = [(observed_q[s][a])**2 for s in range(env.observation_space.n) for a in range(env.action_space.n)]
    return np.sqrt(np.mean(residuals))

if __name__ == "__main__":
    EPISODES = 50
    EPOCHS = 5000

    # find the average error over many runs of the algorithms
    errors_ql = []
    errors_sr = []
    for i in range(EPOCHS):
        q, policy, rmse_ql, policy_list = qlearning(episodes=EPISODES,eps=0.01)
        errors_ql.append(rmse_ql)

        q, policy, rmse_sarsa, policy_list = sarsa(episodes=EPISODES,eps=0.01)
        errors_sr.append(rmse_sarsa)

    mean_ep_error_ql = [np.mean([x[i] for x in errors_ql]) for i in range(EPISODES)]
    mean_ep_error_sr = [np.mean([x[i] for x in errors_sr]) for i in range(EPISODES)]

    plt.plot(mean_ep_error_ql)
    plt.plot(mean_ep_error_sr)
    plt.legend(["QL","SR"])
    plt.show()

    # q, policy, rmse_sarsa, policy_list = sarsa(episodes=EPISODES,eps=0.01)
    # print(policy)
    # plt.plot(rmse_sarsa)
    # plt.legend(["Q Learning","SARSA"])
    # plt.show()