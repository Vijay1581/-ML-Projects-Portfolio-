import numpy as np 
import matplotlib.pyplot as plt 
import random
import warnings
warnings.filterwarnings('ignore')

# Step 1: Environment - Grid World
print("=" * 55)
print("REINFORCEMENT LEARNING - Q-Learning")
print("=" * 55)

class GridWorld:
    """
    5x5 Grid World Environment

    S = Start (0,0) 
    G = Goal  (4,4)
    X = Obstacle
    """
    def __init__(self, size=5):
        self.size        = size
        self.start       = (0, 0)
        self.goal        = (4, 4)
        self.obstacles   = [
            (1, 1), (2, 2), (3, 3),
            (1, 3), (3, 1)
        ]
        self.state       = self.start
        self.actions      = {
            0: (-1, 0),    # Up
            1: (1, 0),     # Down
            2: (0, -1),    # Left
            3: (0, 1)      # Right
        }

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c    = self.state
        dr, dc  = self.actions[action]
        new_r   = max(0, min(
            self.size-1, r+dr
        ))
        new_c   = max(0, min(
            self.size-1, c+dr
        ))

        # Obstacle Check
        if (new_r, new_c) in self.obstacles:
            new_r, new_c = r, c
            reward       = -5
        elif (new_r, new_c) == self.goal:
            reward = 100
            self.state = (new_r, new_c)
            return self.state, reward, True
        else:
            reward = -1

        self.state = (new_r, new_c)
        return self.state, reward, False

    def render(self, q_table=None):
        """Grid Show"""
        grid = [['.' for _ in range(self.size)]
                for _ in range(self.size)]

        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        grid[self.goal[0]][self.goal[1]]   = 'G'
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.state[0]][self.state[1]] = 'A'

        print("\nGrid World:")
        print("+" + "-" * (self.size*2+1) + "+")
        for row in grid:
            print("| " + " ".join(row) + " |")
        print("+" + "-" * (self.size*2+1) + "+")

# Step 3: Q-Learning Agent
print("\n" + "=" * 55)
print("Q-LEARNING AGENT")
print("=" * 55)

class QLearningAgent:
    def __init__(self, state_size,
                 action_size,
                 learning_rate=0.1,
                 discount=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):

        self.action_size   = action_size
        self.lr            = learning_rate
        self.gamma         = discount
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Table: state x action 
        self.q_table = np.zeros(
            (state_size, state_size,
             action_size)
        )

    def get_action(self, state):
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            return random.randint(
                0, self.action_size-1
            )
        return np.argmax(
            self.q_table[state[0], state[1]]
        )

    def update(self, state, action,
               reward, next_state, done):
        # Q-Learning Updates
        current_q = self.q_table[
            state[0], state[1], action
        ]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * \
                np.max(self.q_table[
                    next_state[0],
                    next_state[1]
                ])

        # Update Q-Table
        self.q_table[
            state[0], state[1], action
        ] += self.lr * (target_q - current_q)

        # Decay Epsilon 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_best_path(self, env):
        """Best Path find"""
        state   = env.reset()
        path    = [state]
        visited = set()
        visited.add(state)

        for _ in range(50):
            action = np.argmax(
                self.q_table[state[0], state[1]]
            )
            next_state, _, done = env.step(action)

            if next_state in visited:
                break
            visited.add(next_state)
            path.append(next_state)
            state = next_state

            if done:
                break
        return path

# Step 3: Training
print("\n" + "=" * 55)
print("TRAINING")
print("=" * 55)

env       = GridWorld(size=5)
agent     = QLearningAgent(
    state_size=5, 
    action_size=4,
    epsilon_decay=0.999
)

episodes         = 2000
rewards_history  = []
steps_history    = []
success_history  = []

for episode in range(episodes):
    state        = env.reset()
    total_reward = 0
    steps        = 0
    done         = False

    while not done and steps < 200:
        action                = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action,
                     reward, next_state, done)

        state         = next_state
        total_reward += reward
        steps        += 1

    rewards_history.append(total_reward)
    steps_history.append(steps)
    success_history.append(1 if done else 0)

    if (episode + 1) % 100 == 0:
        recent_success = sum(
            success_history[-100:]
        )
        avg_reward = np.mean(
            rewards_history[-100:]
        )

        print(f"Episode [{episode+1:4d}/{episodes}] "
              f"Avg Reward: {avg_reward:8.2f} "
              f"Success: {recent_success}/100 "
              f"Epsilon: {agent.epsilon:.3f}")

# Step 4: Results
print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)

total_success = sum(success_history)
print(f"Total Success   : {total_success}/{episodes}")
print(f"Success Rate    : {total_success/episodes:.2%}")
print(f"Final Epsilon   : {agent.epsilon:.4f}")
print(f"Final Avg Reward: {np.mean(rewards_history[-100:]):.2f}")

# Best Path
best_path = agent.get_best_path(env)
print(f"\nBest Path     : {best_path}")
print(f"Path Length     : {len(best_path)}")

# Step 5: Visualization 
Fig, axes = plt.subplots(2, 3,
                         figsize=(18, 12))

# Plot 1: Rewards
window = 50
smoothed = np.convolve(
    rewards_history,
    np.ones(window)/window,
    mode='valid'
)
axes[0, 0].plot(rewards_history,
                alpha=0.3, color='blue')
axes[0, 0].plot(smoothed, color='red',
                linewidth=2,
                label=f'Smooth ({window})')
axes[0, 0].set_title('Reward per Episode',
                     fontweight='bold')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Success Rate
window_s = 100
success_smooth = np.convolve(
    success_history,
    np.ones(window_s)/window_s,
    mode='valid'
)
axes[0, 1].plot(success_smooth,
                color='green', linewidth=2)
axes[0, 1].set_title('Success Rate',
                     fontweight='bold')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Success Rate')
axes[0, 1].set_ylim(0, 1)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Steps per Episode
axes[0, 2].plot(steps_history,
                alpha=0.3, color='orange')
steps_smooth = np.convolve(
    steps_history,
    np.ones(window)/window,
    mode='valid'
)
axes[0, 2].plot(steps_smooth,
                color= 'red', linewidth=2,
                label=f'Smooth ({window})')
axes[0, 2].set_title('Steps per Episode',
                     fontweight='bold')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Steps')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Q-Table Heatmap
q_max = np.max(agent.q_table, axis=2)
sns_ax = axes[1, 0]
import seaborn as sns 
sns.heatmap(
    q_max, 
    annot=True, fmt='.1f',
    cmap='YlOrRd',
    ax=sns_ax
)
sns_ax.set_title('Q-Table Max Values',
                 fontweight='bold')
sns_ax.set_xlabel('Column')
sns_ax.set_ylabel('Row')

# Plot 5: Grid World + Path
grid_vis = np.zeros((5, 5))
for obs in env.obstacles:
    grid_vis[obs[0]][obs[1]] = -1
grid_vis[env.goal[0]][env.goal[1]]   = 2
grid_vis[env.start[0]][env.start[1]] = 1

axes[1, 1].imshow(grid_vis, cmap='RdYlGn')

# Path Draw
if len(best_path) > 1:
    path_r = [p[0] for p in best_path]
    path_c = [p[1] for p in best_path]
    axes[1, 1].plot(path_c, path_r,
                    'b-o', linewidth=2,
                    markersize=8,
                    label='Best Path')

axes[1, 1].set_title('Grid World + Best Path',
                     fontweight='bold')
axes[1, 1].legend()

# Labels
for i in range(5):
    for j in range(5):
        if (i, j) in env.obstacles:
            axes[1, 1].text(
                j, i, 'X',
                ha='center', va='center',
                color='white',
                fontweight='bold',
                fontsize=12
            )
        elif (i, j) == env.goal:
            axes[1, 1].text(
                j, i, 'G',
                ha='center', va='center',
                color='white',
                fontweight='bold',
                fontsize=12
            )
        elif (i, j) == env.start:
            axes[1, 1].text(
                j, i, 'S',
                ha='center', va='center',
                color='white',
                fontweight='bold',
                fontsize=12
            )

# Plot 6: Summary
axes[1, 2].axis('off')
summary = f"""
Q-Learning Summary

Environment : Grid World (5x5)
Episodes    : {episodes} 
Actions     : Up, Down, Left, Right

Agent Config:
  Learning Rate  : 0.1
  Discount       : 0.95
  Epsilon Start  : 1.0
  Epsilon End    : {agent.epsilon:.4f}

Results:
  Total Success  : {total_success}/{episodes}
  Success Rate   : {total_success/episodes:.2%}
  Best Path Len  : {len(best_path)}
  Final Reward   : {np.mean(rewards_history[-100:]):.2f}

Best Path: 
  {' → '.join([str(p) for p in best_path])}
"""
axes[1, 2].text(
    0.05, 0.5, summary,
    transform=axes[1, 2].transAxes,
    fontsize=9,
    verticalalignment='center',
    fontfamily='monospace',
    bbox=dict(
        boxstyle='round',
        facecolor='lightcyan',
        alpha=0.8
    )
)
axes[1, 2].set_title('RL Summary',
                     fontweight='bold')
plt.suptitle(
    'Reinforcement Learning - Q-Learning',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('rl_qlearning.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved!")

# Step 6: DQN Introduction 
print("\n" + "=" * 55)
print("DQN - Deep Q-Network")
print("=" * 55)

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size,
                 action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size,
                 action_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=2000)
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.gamma         = 0.95
        self.lr            = 0.001

        self.model = DQN(state_size,
                         action_size)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        self.criterion = nn.MSELoss()

    def remember(self, state, action,
                 reward, next_state, done):
        self.memory.append(
            (state, action, reward,
             next_state, done)
        )

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(
                0, self.action_size-1
            )
        state_t = torch.FloatTensor(state
                                    ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return q_values.argmax().item()

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(
            self.memory, batch_size
        )

        for state, action, reward, \
                next_state, done in batch:
            state_t       = torch.FloatTensor(
                state
            ).unsqueeze(0)
            next_state_t = torch.FloatTensor(
                next_state
            ).unsqueeze(0)

            target = reward
            if not done:
                with torch.no_grad():
                    target += self.gamma * \
                        self.model(
                            next_state_t
                        ).max().item()
           
            current_q = self.model(
            state_t
            )[0][action]
            loss = self.criterion(
                current_q,
                torch.tensor(float(target))
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DQN Test
state_size  = 2
action_size = 4
dqn_agent   = DQNAgent(state_size,
                       action_size)

print(f"DQN Architecture:")
print(dqn_agent.model)
print(f"\nParameters: "
      f"{sum(p.numel() for p in dqn_agent.model.parameters()):,}")

# Final Summary
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"Algorithm           : Q-Learning")
print(f"Environment         : Grid World 5x5")
print(f"Episodes            : {episodes}")
print(f"Success Rate        : {total_success/episodes:.2%}")
print(f"Best Path Len       : {len(best_path)}")
print(f"Final Epsilon       : {agent.epsilon:.4f}")
print(f"\nDQN Model:")
print(f"  Input Size        : {state_size}")
print(f"  Action Size       : {action_size}")
print(f"  Parameters        : "
      f"{sum(p.numel() for p in dqn_agent.model.parameters()):,}")