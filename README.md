# reinforcement-learning-taxi-irl
Inverse Reinforcement Learning using Max Entropy Algorithm Q-learning expert in Taxi environment
# Inverse Reinforcement Learning in Taxi Environment

**Overview**
This project implements Inverse Reinforcement Learning (IRL) in a discrete environment using the Taxi problem from OpenAI Gym. The goal is to model a reward function from expert demonstrations and use it to train an agent capable of reproducing expert-level behavior.

The project follows a full pipeline:
1. Train an expert agent using Q-learning  
2. Collect expert trajectories  
3. Infer the reward function using Maximum Entropy IRL  
4. Recover a policy from the learned reward  

**Problem Description**

In standard Reinforcement Learning, the reward function is known and the agent learns a policy to maximize it. However, in many real-world problems, the reward function is not directly observable.

Inverse Reinforcement Learning (IRL) addresses this by:
- Observing expert behavior  
- Inferring the reward function that explains it  

This project applies IRL to the Taxi-v3 environment, where an agent must:
- Navigate a grid world  
- Pick up a passenger  
- Deliver the passenger to the correct destination  

---

**Methodology**

1. Expert Policy (Q-Learning)
- Implemented Q-learning with epsilon-greedy exploration  
- Used decaying learning rate and exploration schedule  
- Trained over 10,000 episodes  
- Learned a Q-table representing expert behavior  


2. Expert Trajectory Generation
- Generated trajectories using the learned expert policy  
- Stored sequences of (state, action) pairs  
- Used multiple episodes to ensure diversity  


3. Feature Engineering (State-Action Features)
Designed a feature representation for each (state, action):

- Taxi position (row, column)  
- Distance to passenger  
- Distance to destination  
- Step cost (action penalty)  
- Successful interaction (pickup/dropoff indicator)  

This enables reward modeling as:
 R(s, a) = w · φ(s, a)

 
---

4. Maximum Entropy IRL (State-Action Formulation)
Implemented a Maximum Entropy IRL framework:

- Computes expert feature expectations  
- Uses soft value iteration for policy evaluation  
- Learns reward weights via gradient descent  
- Matches expert and learner feature expectations  

---

5. Policy Recovery
- Derived a stochastic policy using softmax over Q-values  
- Compared learned behavior with expert policy  


Results
- The IRL agent successfully approximates expert behavior  
- Learned reward captures meaningful task structure  
- Feature expectation gap decreases during training  
- Learned weights provide interpretable insights into agent strategy  

Example interpretation:
- Positive weight for valid pickup/dropoff actions  
- Negative weight for unnecessary movements (step cost)  

---

**Requirements**

- Python  
- NumPy  
- Gymnasium (Taxi-v3 environment)  
- Matplotlib  
- SciPy  
- Seaborn  
