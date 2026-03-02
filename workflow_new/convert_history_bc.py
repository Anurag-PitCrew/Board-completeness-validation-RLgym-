import json
import numpy as np
from datetime import datetime
from workflow_env import WorkflowEnv

def load_historical_data(filepath="historical_runs.json"):
    """Loads the raw historical runs from your database export."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}.")
        return []

def convert_to_npz(runs, schema_path="workflow_schema.json", output_file="expert_workflow_data.npz"):
    expert_obs = []
    expert_actions = []
    
    # We use the environment to generate the accurate observation matrices
    env = WorkflowEnv(schema_path=schema_path)
    
    successful_runs = 0
    skipped_runs = 0

    print(f"Processing {len(runs)} historical board runs...")

    for run in runs:
        # 1. Reset the environment for a fresh board run
        obs, _ = env.reset()
        
        # 2. Extract and sort executions chronologically
        executions = run.get("task_executions", [])
        
        # Sort by started_at so we replay the actions in the exact order they happened
        try:
            executions.sort(key=lambda x: datetime.fromisoformat(x["started_at"].replace('Z', '+00:00')))
        except Exception as e:
            print(f"Skipping run {run.get('id')} due to timestamp parsing error: {e}")
            skipped_runs += 1
            continue
            
        run_valid = True
        
        # 3. Replay the timeline
        for exec_record in executions:
            task_id = exec_record.get("task_id")
            
            # Find which action index corresponds to this task_id
            action_idx = -1
            for i, task in enumerate(env.task_list):
                if task["id"] == task_id:
                    action_idx = i
                    break
                    
            if action_idx == -1:
                print(f"Warning: Task {task_id} not found in schema. Skipping this run.")
                run_valid = False
                break
                
            # A. Save the State-Action Pair (This is what the BC agent learns from!)
            expert_obs.append(obs.copy())
            expert_actions.append(action_idx)
            
            # B. Step the environment forward to update the state for the next execution
            obs, reward, terminated, _, info = env.step(action_idx)
            
            if "error" in info:
                print(f"Warning: Historical run contained an illegal move: {info['error']}")
                # We might still want to learn from it, but let's log it.
                
            if terminated:
                break # Board is complete
                
        if run_valid:
            successful_runs += 1
        else:
            skipped_runs += 1

    # 4. Save the compiled dataset
    if expert_obs:
        obs_array = np.array(expert_obs)
        act_array = np.array(expert_actions)
        
        np.savez(output_file, obs=obs_array, actions=act_array)
        print("\n--- Conversion Complete ---")
        print(f"Successfully processed {successful_runs} runs.")
        print(f"Skipped {skipped_runs} runs.")
        print(f"Generated {len(obs_array)} total state-action transitions.")
        print(f"Saved to '{output_file}'. Ready for train_workflow_bc.py!")
    else:
        print("\nFailed to generate any valid transitions.")

if __name__ == "__main__":
    # Mocking a small historical run file for the script to execute successfully right away
    mock_history = [
        {
            "id": "run-999",
            "task_executions": [
                {"task_id": "task_1", "started_at": "2026-03-02T10:00:00Z"},
                {"task_id": "task_2", "started_at": "2026-03-02T10:01:00Z"},
                {"task_id": "task_3", "started_at": "2026-03-02T10:02:00Z"},
                {"task_id": "task_4", "started_at": "2026-03-02T10:05:00Z"}
            ]
        }
    ]
    with open("historical_runs.json", "w") as f:
        json.dump(mock_history, f)
        
    runs_data = load_historical_data("historical_runs.json")
    convert_to_npz(runs_data)