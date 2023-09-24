# DQN Agent for CartPole
This is an implementation of a Deep Q-Network (DQN) agent for reinforcement learning tasks. DQN is a popular algorithm used for training agents to make decisions in environments where actions lead to rewards. The agent learns to maximize the cumulative reward it receives over time.

## DQN-Agent
 A simple feedforward neural network with three fully connected layers, that learns to maximize the cumulative reward over time. 

## Replay Buffer
The **ReplayBuffer** class is a dataset of past experiences (state, action, reward, next state) that the agent can use for learning.

## Training
**Create an agent with specified state and action sizes**
```python
agent = Agent(state_size, action_size)
```

**Populate the replay buffer with episodes**
```python
populate_buffer()
```

**Train the agent**
```python
loss = training(agent, replay_buffer, epochs=50, batch_size=10)
```