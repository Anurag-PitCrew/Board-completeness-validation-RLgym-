import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import uuid
import copy

# ==========================================
# 1. Stochastic Task Generator
# ==========================================
class KanbanTaskGenerator:
    def __init__(self, arrival_rate=0.3):
        """
        arrival_rate: Lambda (λ) for Poisson distribution. 
        """
        self.arrival_rate = arrival_rate

    def generate_arrival(self):
        """Returns a list of new tasks based on Poisson arrival."""
        num_new_tasks = np.random.poisson(self.arrival_rate)
        new_tasks = []
        
        for _ in range(num_new_tasks):
            new_tasks.append({
                "id": f"task_{str(uuid.uuid4())[:4]}",
                "priority": random.choice([1, 2, 3]), # 1=High, 3=Low
                "complexity": random.randint(1, 8),   # Story points
                "age_in_days": 0
            })
        return new_tasks

# ==========================================
# 2. Gymnasium Environment
# ==========================================
class KanbanEnv(gym.Env):
    def __init__(self, initial_data=None):
        super(KanbanEnv, self).__init__()
        
        # Default JSON State if none provided
        self.initial_data = initial_data or {
            "board_metadata": {
                "board_id": "dev_sprint_01",
                "wip_limits": {
                    "todo": 99,
                    "in_progress": 3,
                    "review": 2,
                    "done": 99
                }
            },
            "columns": {
                "todo": [
                    {"id": "task_101", "priority": 1, "complexity": 3, "age_in_days": 0},
                    {"id": "task_102", "priority": 2, "complexity": 5, "age_in_days": 1}
                ],
                "in_progress": [
                    {"id": "task_100", "priority": 3, "complexity": 8, "age_in_days": 4}
                ],
                "review": [],
                "done": []
            }
        }
        
        self.max_tasks = 15  # Max capacity of the board for padding
        self.col_order = ["todo", "in_progress", "review", "done"]
        self.generator = KanbanTaskGenerator(arrival_rate=0.2)
        
        # Observation Space: [Task_ID_Proxy, Column_Idx, Priority, Complexity, Age]
        self.observation_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(self.max_tasks, 5), 
            dtype=np.float32
        )
        
        # Action Space: Choose an index (0 to max_tasks - 1) representing the task to move
        self.action_space = spaces.Discrete(self.max_tasks)
        
        self.state_data = copy.deepcopy(self.initial_data)

    def _get_obs(self):
        """Converts JSON structure to a flat NumPy array for the RL agent."""
        obs = np.zeros((self.max_tasks, 5), dtype=np.float32)
        col_map = {name: idx for idx, name in enumerate(self.col_order)}
        
        task_idx = 0
        for col_name in self.col_order:
            for task in self.state_data["columns"][col_name]:
                if task_idx < self.max_tasks:
                    obs[task_idx] = [
                        task_idx,              # ID Proxy
                        col_map[col_name],     # Current Column
                        task["priority"],      # Priority
                        task["complexity"],    # Difficulty
                        task["age_in_days"]    # Latency
                    ]
                    task_idx += 1
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Deep copy to ensure we start fresh every episode
        self.state_data = copy.deepcopy(self.initial_data)
        observation = self._get_obs()
        info = {"wip_limits": self.state_data["board_metadata"]["wip_limits"]}
        return observation, info

    def step(self, action):
        wip_limits = self.state_data["board_metadata"]["wip_limits"]
        
        # 1. Map current tasks exactly as they appear in the observation space
        all_tasks = []
        for col_name in self.col_order:
            for task in self.state_data["columns"][col_name]:
                all_tasks.append((col_name, task))

        # 2. Check if action is valid (agent didn't pick an empty slot)
        if action >= len(all_tasks):
            return self._get_obs(), -1.0, False, False, {"error": "Invalid empty task index"}

        current_col, task_to_move = all_tasks[action]
        current_idx = self.col_order.index(current_col)
        
        # 3. Prevent moving tasks already in 'done'
        if current_idx == len(self.col_order) - 1:
            return self._get_obs(), -2.0, False, False, {"error": "Task already done"}
        
        target_col = self.col_order[current_idx + 1]
        reward = 0.0

        # 4. Enforce WIP Limits & Transition State
        if len(self.state_data["columns"][target_col]) >= wip_limits[target_col]:
            reward = -10.0  # WIP Limit penalty
        else:
            self.state_data["columns"][current_col].remove(task_to_move)
            self.state_data["columns"][target_col].append(task_to_move)
            
            if target_col == "done":
                reward = 20.0  # Task completion reward
            else:
                reward = 1.0   # Progress reward
        
        # 5. Aging Logic
        for col in ["todo", "in_progress", "review"]:
            for t in self.state_data["columns"][col]:
                t["age_in_days"] += 1
                reward -= 0.1  # Global latency penalty

        # 6. Stochastic Arrival
        incoming_tasks = self.generator.generate_arrival()
        for task in incoming_tasks:
            total_tasks = sum(len(tasks) for tasks in self.state_data["columns"].values())
            if total_tasks < self.max_tasks:
                self.state_data["columns"]["todo"].append(task)
        
        # Add a backlog pressure penalty
        backlog_penalty = len(self.state_data["columns"]["todo"]) * -0.05
        reward += backlog_penalty

        # 7. Check termination (Episode ends if 'done' reaches board capacity)
        terminated = len(self.state_data["columns"]["done"]) >= self.max_tasks
        
        return self._get_obs(), float(reward), terminated, False, {}

# ==========================================
# 3. Test Script
# ==========================================
if __name__ == "__main__":
    # Initialize the environment
    env = KanbanEnv()
    obs, info = env.reset()
    
    print("Initial Observation Shape:", obs.shape)
    print("Initial Observation (first 4 rows):\n", obs[:4])
    print("-" * 50)
    
    # Run a few random steps
    for step in range(5):
        # Sample a random action from the action space
        # In a real scenario, your RL model or Behavior Clone would provide this
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step: {step + 1} | Action Taken: {action} | Reward: {reward:.2f}")
        if "error" in info:
            print(f"  -> Note: {info['error']}")
            
        if terminated:
            print("Episode Terminated!")
            break