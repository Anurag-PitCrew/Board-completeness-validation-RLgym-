import numpy as np
import os
from kanban_env import KanbanEnv # Assuming you saved the previous file as kanban_env.py

def expert_policy(obs, env):
    """
    A rule-based expert that mimics optimal Kanban behavior.
    It reads the environment's internal state to make perfect decisions.
    """
    wip_limits = env.state_data["board_metadata"]["wip_limits"]
    cols = env.state_data["columns"]
    
    # 1. Build the exact task list the environment's action space expects
    all_tasks = []
    for col_name in env.col_order:
        for task in cols[col_name]:
            all_tasks.append((col_name, task))
            
    if not all_tasks:
        return 0 # Failsafe if the board is completely empty
        
    # 2. Implement the "Right-to-Left" Pull System strategy
    # We check columns backwards: try to pull into Done, then Review, then In Progress
    pull_priority = [
        ("done", "review"),          # Target: Done, Source: Review
        ("review", "in_progress"),   # Target: Review, Source: In Progress
        ("in_progress", "todo")      # Target: In Progress, Source: Todo
    ]
    
    for target_col, source_col in pull_priority:
        # Check if the target column has room (WIP constraint check)
        if len(cols[target_col]) < wip_limits[target_col]:
            
            # Find all available tasks in the source column
            candidates = []
            for action_idx, (col_name, task) in enumerate(all_tasks):
                if col_name == source_col:
                    candidates.append((action_idx, task))
            
            if candidates:
                # 3. Tie-breaking rules for tasks in the same column:
                # Prioritize 'Priority 1' (High) first, then older tasks.
                candidates.sort(key=lambda x: (x[1]["priority"], -x[1]["age_in_days"]))
                
                # Return the index of the best task to move
                best_action = candidates[0][0]
                return best_action
                
    # 4. If the board is deadlocked (all WIPs full), just pick an action to let time pass
    # (In a real scenario, this shouldn't happen often if WIP limits are balanced)
    return 0 

def generate_expert_dataset(num_episodes=50, max_steps_per_ep=100):
    env = KanbanEnv()
    
    expert_observations = []
    expert_actions = []
    
    print(f"Generating Expert Dataset: {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        for step in range(max_steps_per_ep):
            # Get action from the expert
            action = expert_policy(obs, env)
            
            # Save the State-Action pair
            expert_observations.append(obs.copy())
            expert_actions.append(action)
            
            # Step the environment forward
            obs, reward, terminated, _, info = env.step(action)
            
            if terminated:
                break
                
    # Convert lists to NumPy arrays for training
    obs_data = np.array(expert_observations)
    act_data = np.array(expert_actions)
    
    print(f"Dataset generated! Total steps collected: {len(obs_data)}")
    
    # Save to a compressed NumPy file (.npz)
    save_path = "expert_kanban_data.npz"
    np.savez(save_path, obs=obs_data, actions=act_data)
    print(f"Saved successfully to '{save_path}'")

if __name__ == "__main__":
    generate_expert_dataset()