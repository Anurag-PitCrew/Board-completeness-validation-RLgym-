import numpy as np
from workflow_env import WorkflowEnv

def expert_policy(obs):
    """Finds the first actionable task. Index 4 is the 'Is_Actionable' flag."""
    for task_idx, features in enumerate(obs):
        if features[4] == 1.0: # Actionable
            return task_idx
    return 0 

def generate_data():
    env = WorkflowEnv()
    obs_list, act_list = [], []
    
    print("Generating optimal workflow executions...")
    for _ in range(100): # 100 episodes
        obs, _ = env.reset()
        terminated = False
        
        while not terminated:
            action = expert_policy(obs)
            obs_list.append(obs.copy())
            act_list.append(action)
            
            obs, reward, terminated, _, _ = env.step(action)
            
    np.savez("expert_workflow_data.npz", obs=np.array(obs_list), actions=np.array(act_list))
    print(f"Saved {len(obs_list)} expert transitions!")

if __name__ == "__main__":
    generate_data()