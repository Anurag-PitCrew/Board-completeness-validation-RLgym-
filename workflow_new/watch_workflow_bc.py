import time
import os
import torch
from workflow_env import WorkflowEnv
from train_workflow_bc import WorkflowBCAgent

def watch_bc():
    env = WorkflowEnv()
    
    # Initialize the BC model structure
    model = WorkflowBCAgent(input_dim=50, output_dim=10)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load("workflow_bc.pth"))
        model.eval() # Set to evaluation mode
    except FileNotFoundError:
        print("Error: 'workflow_bc.pth' not found. Run train_workflow_bc.py first.")
        return

    obs, _ = env.reset()
    step = 0
    total_reward = 0
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 70)
        print(f" BC Workflow Agent - Step: {step} | Cumulative Reward: {total_reward:.2f}")
        print("=" * 70)
        
        # Render the board state hierarchically
        for s_idx, swimlane in enumerate(env.state_data["swimlanes"]):
            print(f"\n[Swimlane {s_idx + 1}: {swimlane['name']}]")
            for task in env.task_list:
                if task["swimlane_idx"] == s_idx:
                    if task["status"] == "completed":
                        icon = "✅"
                    elif task["status"] == "failed":
                        icon = "❌"
                    else:
                        icon = "⏳"
                        
                    deps = f"(Deps: {task['dependencies']})" if task['dependencies'] else "(Independent)"
                    print(f"  {icon} [{task['id']}] {task['name']} {deps}")
                    
        if step > 0:
            time.sleep(1.5) # Pause for human readability
            
        # Agent predicts the next best task
        action = model.predict(obs)
        
        # Take the step
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        step += 1
        
        if "error" in info:
            print(f"\n [!] Illegal Move Penalty: {info['error']}")
        elif "success" in info:
            print(f"\n [+] {info['success']}")
            
        if terminated or step >= 20:
            print(f"\nWorkflow Run Finished! Total Steps: {step}, Final Score: {total_reward:.2f}")
            break

if __name__ == "__main__":
    watch_bc()