import time
import os
from stable_baselines3 import PPO
from workflow_env import WorkflowEnv

def watch_ppo():
    env = WorkflowEnv()
    
    print("Loading the trained PPO Workflow model...")
    # Load the final model (or switch to logs_workflow/best_model/best_model)
    model = PPO.load("workflow_ppo_final") 
    
    obs, _ = env.reset()
    step = 0
    total_reward = 0
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 70)
        print(f" RL Workflow Agent (PPO) - Step: {step} | Cumulative Reward: {total_reward:.2f}")
        print("=" * 70)
        
        # Render the board state hierarchically
        for s_idx, swimlane in enumerate(env.state_data["swimlanes"]):
            print(f"\n[Swimlane {s_idx + 1}: {swimlane['name']}]")
            for task in env.task_list:
                if task["swimlane_idx"] == s_idx:
                    # Visual indicators for task status
                    if task["status"] == "completed":
                        icon = "✅"
                    elif task["status"] == "failed":
                        icon = "❌"
                    else:
                        icon = "⏳"
                        
                    deps = f"(Deps: {task['dependencies']})" if task['dependencies'] else "(Independent)"
                    print(f"  {icon} [{task['id']}] {task['name']} {deps}")
                    
        if step > 0:
            time.sleep(1.5) # Pause to let the human eye track the changes
            
        # PPO predicts the best task to execute
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Print what the agent just did
        if "error" in info:
            print(f"\n [!] Illegal Move Penalty: {info['error']}")
        elif "success" in info:
            print(f"\n [+] {info['success']}")
            
        if terminated or step >= 20:
            print(f"\nWorkflow Run Finished! Total Steps: {step}, Final Score: {total_reward:.2f}")
            break

if __name__ == "__main__":
    watch_ppo()