import time
import os
from stable_baselines3 import PPO
from kanban_env import KanbanEnv

def clear_screen():
    """Clears the terminal screen for a smooth animation effect."""
    os.system('cls' if os.name == 'nt' else 'clear')

def render_board(env, step, reward, action_taken=None):
    """Prints a visual representation of the JSON board state."""
    clear_screen()
    state = env.state_data
    wip_limits = state["board_metadata"]["wip_limits"]
    
    print("=" * 70)
    print(f" PPO Kanban Agent - Step: {step} | Cumulative Reward: {reward:.2f}")
    if action_taken is not None:
        print(f" Last Action Taken: Selected Task Index {action_taken}")
    print("=" * 70)
    
    # Define column formatting
    col_order = ["todo", "in_progress", "review", "done"]
    col_widths = 16
    
    # Print Headers with WIP limits
    headers = ""
    for col in col_order:
        wip = wip_limits[col]
        wip_str = str(wip) if wip < 99 else "∞"
        title = f"{col.upper()} ({wip_str})"
        headers += f"{title:<{col_widths}}| "
    print(headers)
    print("-" * 70)
    
    # Find the maximum number of tasks in any column to know how many rows to print
    max_rows = max([len(state["columns"][col]) for col in col_order])
    
    if max_rows == 0:
        print("Board is completely empty!")
        
    # Print row by row
    for row in range(max_rows):
        row_str = ""
        for col in col_order:
            tasks = state["columns"][col]
            if row < len(tasks):
                # Format: ID (Pri:1, Age:2)
                t = tasks[row]
                task_str = f"{t['id'][5:]} (P{t['priority']}, {t['age_in_days']}d)"
                row_str += f"{task_str:<{col_widths}}| "
            else:
                # Empty slot
                row_str += f"{'':<{col_widths}}| "
        print(row_str)
        
    print("=" * 70)
    print("Press Ctrl+C to stop watching.\n")

def watch_ppo():
    env = KanbanEnv()
    
    # 1. Load the trained PPO model from the zip file
    print("Loading the trained PPO model...")
    # Note: If your best model performed better than the final one, 
    # you can change this to load "./logs/best_model/best_model"
    model = PPO.load("kanban_ppo_final") 
    
    # 2. Reset the environment for a new episode
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    
    render_board(env, step, total_reward)
    time.sleep(2)
    
    # 3. Run the episode loop
    while True:
        step += 1
        
        # The PPO model returns the action and the state. 
        # `deterministic=True` tells the agent to pick its absolute best move, no guessing.
        action, _states = model.predict(obs, deterministic=True)
        
        # SB3 sometimes returns actions as NumPy arrays, so we cast to int
        action = int(action) 
        
        # Take the step
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        
        # Draw the updated board
        render_board(env, step, total_reward, action_taken=action)
        
        if "error" in info:
            print(f" [!] Penalty: {info['error']}")
            
        time.sleep(1.0) # Pause for 1 second so you can read the board
        
        # Stop after 50 steps so it doesn't run forever
        if terminated or step >= 50:
            print(f"\nEpisode Finished! Total Steps: {step}, Final Score: {total_reward:.2f}")
            break

if __name__ == "__main__":
    watch_ppo()