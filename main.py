import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


env = gym.make("Taxi-v3")

def epsilon_choice(epsilon):
    # 0 - exploration 1 - exploatation
    p = random.random()
    if p < epsilon:
        return 0
    else:
        return 1


def Q_learning_episode(env, Q, gamma, alpha, epsilon):
    # 1. Initialization, env reset gives a touple
    current_state_tuple, info = env.reset()

    # Ensure current_state is a simple integer
    current_state = int(current_state_tuple)

    action_list = [0,1,2,3,4,5]
    terminated = False
    truncated = False
    total_reward = 0

    while (not terminated) and (not truncated):

        choice = epsilon_choice(epsilon)

        # Action Selection
        if choice == 1:
            # Exploitation (tie-breaking)
            max_q = np.max(Q[current_state])
            max_indices = np.where(Q[current_state] == max_q)[0]
            action = np.random.choice(max_indices)
        else:
            # Exploration
            action = random.choice(action_list)

        # 2. Environment Step
        next_state_tuple, reward, terminated, truncated, info = env.step(action)

        # Ensure next_state is a simple integer index
        next_state = int(next_state_tuple)
        total_reward += reward

        # 3. Q-Learning Update
        # Note: Use np.max() for NumPy array maximum
        max_next_q = np.max(Q[next_state])

        Q[current_state, action] += alpha * (reward + gamma * max_next_q - Q[current_state, action])

        # 4. State Transition
        current_state = next_state

    return Q, total_reward


def Q_learning(env, Q, gamma, alpha, epsilon_start, epsilon_min, episode_n):

    reward_history = []

    for i in range(episode_n):

        alpha_t = alpha / (1 + 0.0001 * i)
        epsilon = epsilon_min + (epsilon_start - epsilon_min)* np.exp(-0.0005 * i)
        Q, episode_reward = Q_learning_episode(env, Q, gamma, alpha_t, epsilon)
        reward_history.append(episode_reward)

    return Q, reward_history

# 2. Reset the environment to get the initial state and information
# The underscore is used to discard the 'info' dictionary, which you may not need immediately.
initial_state, info = env.reset()

S = 500
A = 6
gamma = 0.95
alpha = 0.2
epsilon_start = 1
epsilon_min = 0.05
episode_n = 10000
Q = np.zeros((S,A))

print(f"Initial State: {initial_state}")
print(f"Observation Space (States): {env.observation_space.n}")
print(f"Action Space (Actions): {env.action_space.n}")
Q_expert, reward_history = Q_learning(env,Q,gamma,alpha, epsilon_start, epsilon_min, episode_n)
print(Q[470])

def moving_average(x, window=100):
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='valid')

episodes = np.arange(len(reward_history))
rewards_ma = moving_average(reward_history, window=100)

# plt.figure(figsize=(10, 6))

# # Raw episode rewards (light)
# plt.plot(episodes, reward_history, color='gray', alpha=0.3, label='Reward (raw)')

# # Smoothed rewards
# plt.plot(np.arange(len(rewards_ma)), rewards_ma,
#          color='blue', label='Reward (100-episode moving average)')

# plt.xlabel('Episode')
# plt.ylabel('Total reward per episode')
# plt.title('Learning curve: Taxi-v3 Q-learning agent')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

def generate_expert_trajectories(env, Q, n_episodes=50, max_steps=200):
    trajectories = []  # list of episodes; each episode is list of (s, a)

    for i in range(n_episodes):
        s, info = env.reset()
        s = int(s)
        episode = []

        for t in range(max_steps):
            # Greedy expert policy from Q
            a = int(np.argmax(Q[s]))

            s_next, r, terminated, truncated, info = env.step(a)
            s_next = int(s_next)
            done = terminated or truncated

            episode.append((s, a))

            s = s_next
            if done:
                break

        trajectories.append(episode)

    return trajectories
#use longer episodes if needed
expert_trajectories = generate_expert_trajectories(env, Q_expert, n_episodes=1000, max_steps=200)

# Save trajectorie to disk for IRL
# with open(r"C:\Users\Dimitrison\PycharmProjects\PythonProjectaxi\trajectories.pkl", "wb") as f:
#     pickle.dump(expert_trajectories, f)

########### INVERSE REINFCORCEMENT LEARNING #################################

# --- 1. Feature Construction (State-Action) ---
def build_sa_features(env):
    S = env.unwrapped.observation_space.n
    A = env.unwrapped.action_space.n
    # Features: 
    # 0: Row (Normalized)
    # 1: Col (Normalized)
    # 2: Dist to Passenger (Linear)
    # 3: Dist to Destination (Linear)
    # 4: Action Bias (Step Cost)
    # 5: Successful Interaction (Pickup/Dropoff) - NEW!
    F = 6
    
    # Shape is now (S, A, F)
    features = np.zeros((S, A, F))
    
    MAX_DIST = 10.0
    
    for s in range(S):
        taxi_row, taxi_col, passenger_loc, dest = env.unwrapped.decode(s)
        
        # Base state features (shared across all actions for this state)
        f_row = taxi_row / 4.0
        f_col = taxi_col / 4.0
        
        # Distance to Passenger
        if passenger_loc < 4:
            p_locs = [(0,0), (0,4), (4,0), (4,3)]
            pr, pc = p_locs[passenger_loc]
            dist = abs(taxi_row - pr) + abs(taxi_col - pc)
            f_p_dist = (MAX_DIST - dist) / MAX_DIST
        else:
            f_p_dist = 0.0

        # Distance to Destination
        d_locs = [(0,0), (0,4), (4,0), (4,3)]
        dr, dc = d_locs[dest]
        if passenger_loc == 4: # In taxi
            dist = abs(taxi_row - dr) + abs(taxi_col - dc)
            f_d_dist = (MAX_DIST - dist) / MAX_DIST
        else:
            f_d_dist = 0.0
            
        # Assign these base features to all actions first
        features[s, :, 0] = f_row
        features[s, :, 1] = f_col
        features[s, :, 2] = f_p_dist
        features[s, :, 3] = f_d_dist
        features[s, :, 4] = 1.0 # Step cost applies to all actions
        
        # --- ACTION SPECIFIC LOGIC ---
        # We need to simulate the transition to see if an action is valid/useful
        # Actions: 0:S, 1:N, 2:E, 3:W, 4:Pickup, 5:Dropoff
        
        for a in range(A):
            # Check for Successful Interaction
            # Logic: If we are at passenger and action is Pickup (4)
            is_valid_pickup = (passenger_loc < 4) and \
                              (taxi_row == p_locs[passenger_loc][0]) and \
                              (taxi_col == p_locs[passenger_loc][1]) and \
                              (a == 4)
            
            # Logic: If we are at dest, pass is in taxi, and action is Dropoff (5)
            is_valid_dropoff = (passenger_loc == 4) and \
                               (taxi_row == d_locs[dest][0]) and \
                               (taxi_col == d_locs[dest][1]) and \
                               (a == 5)
            
            if is_valid_pickup or is_valid_dropoff:
                features[s, a, 5] = 1.0
            
    return features


# --- 2. Corrected Expert Expectations (State-Action) ---
def compute_expert_sa_expectations(trajectories, feature_matrix, gamma):
    S, A, F = feature_matrix.shape
    mu = np.zeros(F)

    for traj in trajectories:
        for t, (s, a) in enumerate(traj):
            # Now we look up features using [s, a]
            mu += (gamma ** t) * feature_matrix[s, a]

    mu /= len(trajectories)
    return mu


# --- 3. Soft Value Iteration with R(s,a) ---
def soft_value_iteration_sa(env, R_sa, gamma=0.95, iterations=100):
    # R_sa shape is (S, A)
    uenv = env.unwrapped
    S = uenv.observation_space.n
    A = uenv.action_space.n
    P = uenv.P

    V = np.zeros(S)

    for _ in range(iterations):
        Q = np.zeros((S, A))
        for s in range(S):
            for a in range(A):
                val_next = 0
                for prob, s_next, r, done in P[s][a]:
                    if not done:
                        val_next += prob * V[s_next]
                
                # CRITICAL CHANGE: R depends on 'a' now
                Q[s, a] = R_sa[s, a] + gamma * val_next
        
        # LogSumExp for numerical stability
        max_q = np.max(Q, axis=1)
        V = max_q + np.log(np.sum(np.exp(Q - max_q[:, None]), axis=1))

    return V

def compute_soft_policy_sa(env, R_sa, V, gamma=0.95):
    uenv = env.unwrapped
    S = uenv.observation_space.n
    A = uenv.action_space.n
    P = uenv.P

    Q = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            # Same R(s,a) update here
            Q[s, a] = R_sa[s, a] + gamma * sum(
                prob * V[s_next]
                for prob, s_next, r, done in P[s][a]
            )

    pi = np.exp(Q - V[:, None])
    return pi


# --- 4. Learner Expectations (State-Action Visitation) ---
def compute_expected_sa_counts(env, pi, features, expert_trajectories, gamma=0.95, horizon=200):
    uenv = env.unwrapped
    S = uenv.observation_space.n
    A = uenv.action_space.n
    F = features.shape[2]
    P = uenv.P
    
    # We need to track D[s, a], not just D[s]
    D_sa = np.zeros((S, A))
    
    # Start distribution
    start_counts = np.zeros(S)
    for traj in expert_trajectories:
        start_counts[traj[0][0]] += 1
    p_active = start_counts / len(expert_trajectories)

    for t in range(horizon):
        p_next_active = np.zeros(S)
        
        # Iterate over active states
        for s in np.where(p_active > 1e-5)[0]:
            mass = p_active[s]
            
            for a in range(A):
                prob_a = pi[s, a]
                if prob_a < 1e-5: continue
                
                flow_sa = mass * prob_a
                
                # Add to state-action visitation count
                D_sa[s, a] += (gamma ** t) * flow_sa
                
                # Transition
                for trans_prob, next_s, reward, done in P[s][a]:
                    if not done:
                        p_next_active[next_s] += flow_sa * trans_prob
                    # If done, the flow stops here (we don't add to p_next)

        p_active = p_next_active
        if np.sum(p_active) < 1e-5:
            break

    # Final dot product: Sum over all (s, a) of (Count[s,a] * Features[s,a])
    # D_sa is (S, A), features is (S, A, F)
    # Tensordot sums over the first 2 axes (S, A)
    return np.tensordot(D_sa, features, axes=([0,1], [0,1]))


# --- 5. Main MaxEnt Loop (Updated) ---
def maxent_irl_sa(env, expert_trajectories, gamma=0.99, lr=0.1, iters=100):
    print("Building (S,A) features...")
    features = build_sa_features(env) # Now (S, A, F)
    
    print("Computing expert expectations...")
    expert_mu = compute_expert_sa_expectations(expert_trajectories, features, gamma)
    
    w = np.zeros(features.shape[2])
    
    for it in range(iters):
        # 1. Calculate R(s,a)
        # features (S,A,F) @ w (F,) -> R (S,A)
        R_sa = np.dot(features, w)
        
        # 2. Solve
        V = soft_value_iteration_sa(env, R_sa, gamma)
        pi = compute_soft_policy_sa(env, R_sa, V, gamma)
        
        # 3. Learner Expectations
        learner_mu = compute_expected_sa_counts(env, pi, features, expert_trajectories, gamma)
        
        # 4. Update
        grad = expert_mu - learner_mu
        w += lr * grad
        
        if it % 10 == 0:
            diff = np.linalg.norm(grad)
            print(f"Iter {it}: Grad Norm {diff:.4f}")

    return w, features, learner_mu

# 1. Build the NEW State-Action features (Size = 6)
print("Building (S,A) features...")
features_sa = build_sa_features(env)

# 2. Recalculate Expert Expectations using these new features
print("Computing expert expectations for State-Action pairs...")
# CRITICAL: Use the new function 'compute_expert_sa_expectations'
expert_mu_sa = compute_expert_sa_expectations(expert_trajectories, features_sa, gamma=0.95)

# 3. Run the MaxEnt IRL (State-Action version)
print("Running MaxEnt IRL (State-Action)...")
w_sa, _, learner_mu_sa = maxent_irl_sa(env, expert_trajectories, gamma=0.95, lr=0.1, iters=500)
R_learned = np.dot(features_sa, w_sa)

# 4. Verification (The code you wanted to use)
print("\nFinal Feature Expectation Gap:")
print("Expert:", expert_mu_sa)
print("Learner:", learner_mu_sa)
print("Gap:", expert_mu_sa - learner_mu_sa)

# Optional: Print the learned weights to interpret the strategy
print("\nLearned Weights:")
labels = ["Row", "Col", "Dist_Pass", "Dist_Dest", "Step_Cost", "Interaction"]
for label, val in zip(labels, w_sa):
    print(f"{label}: {val:.4f}")

# Example: Check rewards for State 0 (Passenger at R, Dest at G, Taxi at R)
s = 0 
# Recall: Action 4 is Pickup, Action 5 is Dropoff

print("Reward for Moving South:", R_learned[s, 0])   # Should be negative (Step Cost)
print("Reward for Pickup:",       R_learned[s, 4])   # Should be POSITIVE (Interaction + Step Cost)
print("Reward for Dropoff:",      R_learned[s, 5])   # Should be negative (Illegal Dropoff -> No Interaction Reward)

