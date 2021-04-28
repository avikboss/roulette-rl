import gym
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from collections import defaultdict

def qlearning(episodes=2000, alpha=0.01, gamma=0.9, eps=0.1):

    env = gym.make("Roulette-v0")
    env.reset()
    q = defaultdict(lambda: defaultdict(int))
    rmse_list = []
    policy_list = []

    for _ in range(episodes):
        print(f"Episode {_+1} of {episodes}...",end="\r")

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
    q, policy, rmse_ql, policy_list = qlearning(episodes=10,eps=0.01)
    print(policy)
    plt.plot(policy_list)
    plt.show()