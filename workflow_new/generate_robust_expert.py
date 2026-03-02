import numpy as np
import random
import copy
from workflow_env import WorkflowEnv

# ==========================================
# 1. Procedural DAG Generator
# ==========================================
def generate_random_board():
    """Generates a highly randomized but mathematically valid DAG workflow."""
    task_types = ["llm_call", "tool_call", "human_review", "conditional"]
    num_swimlanes = random.randint(1, 3)
    
    board = {
        "board": {"name": "Procedural Training Board"},
        "swimlanes": []
    }
    
    global_task_counter = 1
    all_previous_task_ids = []
    
    for s_idx in range(num_swimlanes):
        num_tasks = random.randint(2, 5) # 2 to 5 tasks per swimlane
        swimlane = {
            "name": f"Swimlane_{s_idx + 1}",
            "position": s_idx + 1,
            "tasks": []
        }
        
        for _ in range(num_tasks):
            task_id = f"task_{global_task_counter}"
            
            # Decide dependencies (Can only depend on tasks that already exist to ensure a valid DAG)
            deps = []
            if all_previous_task_ids:
                # 50% chance to be independent, 50% chance to have 1-2 dependencies
                if random.random() > 0.5:
                    num_deps = random.randint(1, min(2, len(all_previous_task_ids)))
                    deps = random.sample(all_previous_task_ids, num_deps)
            
            task = {
                "id": task_id,
                "name": f"Auto Task {global_task_counter}",
                "task_type": random.choice(task_types),
                "status": "pending",
                "dependencies": deps
            }
            
            swimlane["tasks"].append(task)
            all_previous_task_ids.append(task_id)
            global_task_counter += 1
            
        board["swimlanes"].append(swimlane)
        
    return board

# ==========================================
# 2. Robust Expert Policy (Variance + Heuristics)
# ==========================================
def robust_expert_policy(obs):
    actionable_tasks = []
    
    for task_idx, features in enumerate(obs):
        if features[4] == 1.0: # is_actionable
            actionable_tasks.append({
                "index": task_idx,
                "task_type": features[1] 
            })
            
    if not actionable_tasks:
        return 0 

    # Shuffle to create variance (teach the AI parallel options are valid)
    random.shuffle(actionable_tasks)

    # Sort so 'tool_calls' (1) are prioritized over 'llm_calls' (0) and 'human' (2)
    actionable_tasks.sort(key=lambda x: x["task_type"] == 1, reverse=True) 

    return actionable_tasks[0]["index"]

# ==========================================
# 3. Mass Generation Loop
# ==========================================
def generate_robust_data(num_boards=500):
    obs_list, act_list = [], []
    
    print(f"Generating {num_boards} entirely unique board schemas...")
    
    for episode in range(num_boards):
        # 1. Generate a brand new, random board architecture
        random_schema = generate_random_board()
        
        # 2. Load it into a fresh environment
        env = WorkflowEnv(schema_data=random_schema)
        obs, _ = env.reset()
        
        terminated = False
        step_safeguard = 0
        
        # 3. Let the expert solve the random board
        while not terminated and step_safeguard < env.max_tasks:
            action = robust_expert_policy(obs)
            
            obs_list.append(obs.copy())
            act_list.append(action)
            
            obs, reward, terminated, _, info = env.step(action)
            step_safeguard += 1
            
        if (episode + 1) % 100 == 0:
            print(f" -> Solved {episode + 1} distinct workflows...")
            
    np.savez("expert_workflow_data.npz", obs=np.array(obs_list), actions=np.array(act_list))
    print("\n[SUCCESS] Robust Dataset Generated!")
    print(f"Total state-action transitions captured: {len(obs_list)}")

if __name__ == "__main__":
    generate_robust_data()