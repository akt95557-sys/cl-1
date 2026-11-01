
import numpy as np
import random

# 0 = Free path, 1 = Wall
maze = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
]

start = (0, 0)
goal = (3, 3)

actions = ['U', 'D', 'L', 'R']
rows, cols = 4, 4

# Create Q-table
Q = np.zeros((rows, cols, 4))

alpha = 0.7
gamma = 0.9
epsilon = 0.9
episodes = 500

# Environment step function
def step(state, action):
    r, c = state

    if action == 'U': r -= 1
    elif action == 'D': r += 1
    elif action == 'L': c -= 1
    elif action == 'R': c += 1

    if (r < 0 or r >= rows or c < 0 or c >= cols
            or maze[r][c] == 1):
        return state, -1

    if (r, c) == goal:
        return (r, c), 10

    return (r, c), -0.1

# Training the agent
for episode in range(episodes):
    state = start

    while state != goal:
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = actions[np.argmax(Q[state])]

        next_state, reward = step(state, action)
        a = actions.index(action)

        Q[state[0], state[1], a] += alpha * (
            reward + gamma * np.max(Q[next_state])
            - Q[state[0], state[1], a]
        )

        state = next_state

# Test the learned path
state = start
path = [state]

while state != goal:
    action = actions[np.argmax(Q[state])]
    state, reward = step(state, action)
    path.append(state)

print("Learned Path:")
print(path)
