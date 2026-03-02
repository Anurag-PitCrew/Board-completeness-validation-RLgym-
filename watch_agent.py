import time
import os
import torch
from kanban_env import KanbanEnv
from train_bc import KanbanBCAgent # Imports the brain structure we built

def clear_screen():
    """Clears the terminal screen for a smooth animation effect."""
    os.system('cls' if os.name == 'nt' else 'clear')

def render_board(env, step, reward, action_taken=None):
    """Prints a visual representation of the JSON board state."""
    clear_screen()
    state = env.state_data
    wip_limits = state["board_metadata"]["wip_limits"]
    
    print("=" * 70)
    print(f" Kanban RL Agent - Step: {step} | Cumulative Reward: {reward:.2f}")
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

def watch():
    # 1. Initialize environment
    env = KanbanEnv()
    
    # 2. Load the trained model
    input_dim = 15 * 5
    output_dim = 15
    model = KanbanBCAgent(input_dim, output_dim)
    
    try:
        model.load_state_dict(torch.load("kanban_bc_model.pth"))
        model.eval() # Set to evaluation mode
    except FileNotFoundError:
        print("Error: 'kanban_bc_model.pth' not found. Did you run train_bc.py first?")
        return

    # 3. Run a visual episode
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    
    # Initial render
    render_board(env, step, total_reward)
    time.sleep(2)
    
    while True:
        step += 1
        
        # Agent predicts the action
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits = model(obs_tensor)
            action = torch.argmax(logits, dim=1).item()
            
        # Take the step in the environment
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        
        # Draw the updated board
        render_board(env, step, total_reward, action_taken=action)
        
        # If the agent made an illegal move (e.g. WIP violation), print the warning
        if "error" in info:
            print(f" [!] Penalty: {info['error']}")
            
        # Pause so the human eye can track the movement
        time.sleep(1.5)
        
        if terminated or step >= 50:
            print(f"\nEpisode Finished! Total Steps: {step}, Final Score: {total_reward:.2f}")
            break

if __name__ == "__main__":
    watch()