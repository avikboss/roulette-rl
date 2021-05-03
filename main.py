import gym
import numpy as np
import time
from copy import deepcopy
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd

def sarsa(episodes=2000, alpha=0.01, gamma=0.9, eps=0.1):
    env = gym.make("Roulette-v0")
    env.reset()
    q = defaultdict(lambda: defaultdict(int))
    rmse_list = []
    policy_list = []
    total_steps = 0
    step_list = []
    start = time.time()

    for i in range(episodes):
        s = env.observation_space.sample()
        policy = defaultdict(int)
        a = get_eps_greedy_action(env, eps, policy)
        done = False

        while not done:
            total_steps += 1
            next_s, reward, done, _ = env.step(a)
            next_a = get_eps_greedy_action(env, eps, policy)
            q[s][a] = q[s][a] + alpha * (reward + gamma*q[next_s][next_a] - q[s][a])
            policy[s] = max(q[s], key=q[s].get)
            s = next_s
            a = next_a
        rmse_list.append(rmse(q,env))
        policy_list.append(policy[0])
        step_list.append(total_steps)
        total_steps = 0
        env.reset()

    policy = {s:max(q[s], key=q[s].get) for s in range(env.observation_space.n)}
    end = time.time()
    total_time = end - start
    return q, policy, rmse_list, policy_list, total_time, total_steps, step_list

def qlearning(episodes=2000, alpha=0.01, gamma=0.9, eps=0.1):

    env = gym.make("Roulette-v0")
    env.reset()
    q = defaultdict(lambda: defaultdict(int))
    rmse_list = []
    policy_list = []
    total_steps = 0
    step_list = []
    start = time.time()

    for _ in range(episodes):

        s = env.observation_space.sample()
        policy = defaultdict(int)

        done = False
        while not done:
            total_steps += 1
            a = get_eps_greedy_action(env,eps, policy)
            next_s, reward, done, _ = env.step(a)
            try:
                best_a = max(q[next_s],key=q[next_s].get)
            except:
                best_a = env.action_space.sample()
            q[s][a] = q[s][a] + alpha * (reward + gamma*q[next_s][best_a] - q[s][a])
            policy[s] = max(q[s], key=q[s].get)

            s = next_s

        rmse_list.append(rmse(q,env))
        policy_list.append(policy[0])
        step_list.append(total_steps)
        total_steps = 0
        env.reset()

    policy = {s:max(q[s], key=q[s].get) for s in q.keys()}
    end = time.time()
    total_time = end - start
    return q, policy, rmse_list, policy_list, total_time, total_steps, step_list

def montecarlo_es(episodes, eps):
#     Initialize, for all s ∈ S, a ∈ A(s):
        # Q(s, a) ← arbitrary
        # π(s) ← arbitrary
        # Returns(s, a) ← empty list
#     Repeat forever:
        # Choose S0 ∈ S and A0 ∈ A(S0) s.t. all pairs have probability > 0
        # Generate an episode starting from S0, A0, following π
        # For each pair s, a appearing in the episode:
        # G ← return following the first occurrence of s, a
        # Append G to Returns(s, a)
        # Q(s, a) ← average(Returns(s, a))
        # For each s in the episode:
        # π(s) ← argmaxa Q(s, a)

    env = gym.make("Roulette-v0")
    env.reset()

    # initialize q, policy returns
    q = {0:{a:0 for a in range(env.action_space.n)}}
    policy = {0:0}
    returns = {a:[] for a in range(env.action_space.n)}
    rmse_list = []
    policy_list = []
    step_list = []
    
    for _ in range(episodes):
        
        num_steps = 0
        a = env.action_space.sample() # exploring starts
        taken_actions = []
        done = False

        while not done:
            num_steps += 1
            next_s, reward, done, _ = env.step(a) # simulate an episode, recording the first returns of each action
            if a not in taken_actions:
                taken_actions.append(a)
                returns[a].append(reward)
            a = get_eps_greedy_action(env,eps,policy)

        q[0].update({a:np.mean(returns[a]) for a in taken_actions}) # at the end of an episode, update q and policy
        policy[0] = max(q[0],key=q[0].get)
        rmse_list.append(rmse(q,env))
        policy_list.append(policy[0])
        env.reset()
        step_list.append(num_steps)
        

    return q, policy, rmse_list, policy_list, step_list

def get_eps_greedy_action(env,eps,policy):
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return policy[0]

def rmse(observed_q, env):
    expected_q = {0:{a:0 for a in range(env.action_space.n)}}
    expected_q[0].update({37:0,0:-1/37})
    residuals = [(observed_q[s][a]-expected_q[s][a])**2 for s in expected_q.keys() for a in expected_q[0].keys()]
    return np.sqrt(np.mean(residuals))

if __name__ == "__main__":
    
    EPISODES = 5000
    EPOCHS = 50

    # q, policy, rmse_ql, policy_list = qlearning(episodes=EPISODES,eps=0.01)
    # print(policy)

    # find the average error over many runs of the algorithms
    errors_ql = []
    errors_sr = []
    errors_mc = []
    runtime_ql = []
    runtime_sr = []
    num_steps_ql = []
    num_steps_sr = []
    num_steps_mc = []

    for i in range(EPOCHS):
        q, policy, rmse_ql, policy_list, runtime, num_steps, step_list = qlearning(episodes=EPISODES,eps=0.01)
        errors_ql.append(rmse_ql)
        runtime_ql.append(runtime)
        num_steps_ql.append(step_list)

        q, policy, rmse_sarsa, policy_list, runtime, num_steps, step_list = sarsa(episodes=EPISODES,eps=0.01)
        errors_sr.append(rmse_sarsa)
        runtime_sr.append(runtime)
        num_steps_sr.append(step_list)

        q, policy, rmse_mc, policy_list, step_list = montecarlo_es(episodes=EPISODES,eps=0.01)
        errors_mc.append(rmse_mc)
        num_steps_mc.append(step_list)

    mean_ep_error_ql = [np.mean([x[i] for x in errors_ql]) for i in range(EPISODES)]
    mean_ep_error_sr = [np.mean([x[i] for x in errors_sr]) for i in range(EPISODES)]
    mean_ep_error_mc = [np.mean([x[i] for x in errors_mc]) for i in range(EPISODES)]

    mean_ep_steps_ql = [np.mean([x[i] for x in num_steps_ql]) for i in range(EPISODES)]
    mean_ep_steps_sr = [np.mean([x[i] for x in num_steps_sr]) for i in range(EPISODES)]
    mean_ep_steps_mc = [np.mean([x[i] for x in num_steps_mc]) for i in range(EPISODES)]


    q, policy, rmse_mc, policy_list, step_list = montecarlo_es(episodes=EPISODES,eps=0.01)
    print(policy_list)
    plt.plot(step_list)
    plt.show()
    print("Mean Error:")
    plt.plot(mean_ep_error_ql)
    plt.plot(mean_ep_error_sr)
    plt.plot(mean_ep_error_mc)
    plt.legend(["QL","SR","MC"])
    plt.show()

    print("Runtimes:")
    pd.DataFrame(runtime_ql).plot(kind="density")
    pd.DataFrame(runtime_sr).plot(kind="density")
    plt.show()
    plt.plot(runtime_ql)
    plt.plot(runtime_sr)
    plt.legend(["QL", "SR"])
    plt.show()
    print("Average QL Runtime:", sum(runtime_ql)/len(runtime_ql))
    print("Average SR Runtime:", sum(runtime_sr)/len(runtime_sr))

    print("Number of Steps:")
    plt.plot(mean_ep_steps_ql)
    plt.plot(mean_ep_steps_sr)
    plt.plot(mean_ep_steps_mc)
    plt.legend(["QL", "SR","MC"])
    plt.show()
    print("Average QL Steps:", sum(num_steps_ql)/len(num_steps_ql))
    print("Average SR Steps:", sum(num_steps_sr)/len(num_steps_sr))
