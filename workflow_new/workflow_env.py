import os
import json  # <--- Moved to the top!
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class WorkflowEnv(gym.Env):
    def __init__(self, schema_data=None, schema_path=os.path.join(CURRENT_DIR, "workflow_schema.json")):
        super(WorkflowEnv, self).__init__()
        
        # 1. Accept raw dict if provided (for procedural data generation)
        if schema_data is not None:
            # Deep copy to ensure we don't accidentally modify the source dictionary
            self.initial_data = copy.deepcopy(schema_data) 
        # 2. Otherwise, load from the default JSON file
        else:
            with open(schema_path, 'r') as f:
                self.initial_data = json.load(f)
                
        self.max_tasks = 20 # Max tasks the board can hold
        
        # Observation Space: [Swimlane_Idx, Task_Type, Status, Unmet_Dependencies, Is_Actionable]
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(self.max_tasks, 5), dtype=np.float32
        )
        
        # Action Space: Pick a task ID to execute
        self.action_space = spaces.Discrete(self.max_tasks)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_data = copy.deepcopy(self.initial_data)
        
        # Flatten tasks for easy index mapping
        self.task_list = []
        for s_idx, swimlane in enumerate(self.state_data["swimlanes"]):
            for task in swimlane["tasks"]:
                task["swimlane_idx"] = s_idx
                self.task_list.append(task)
                
        return self._get_obs(), {}

    def _get_unmet_deps(self, task):
        count = 0
        for dep_id in task.get("dependencies", []):
            target = next((t for t in self.task_list if t["id"] == dep_id), None)
            if target and target["status"] != "completed":
                count += 1
        return count

    def _get_obs(self):
        obs = np.zeros((self.max_tasks, 5), dtype=np.float32)
        type_map = {"llm_call": 0, "tool_call": 1, "human_review": 2, "conditional": 3}
        status_map = {"pending": 0, "running": 1, "completed": 2, "failed": 3}
        
        for idx, task in enumerate(self.task_list):
            if idx < self.max_tasks:
                unmet_deps = self._get_unmet_deps(task)
                # Task is actionable if it's pending and has 0 unmet dependencies
                is_act = 1 if (unmet_deps == 0 and task["status"] == "pending") else 0
                
                obs[idx] = [
                    task["swimlane_idx"],
                    type_map.get(task.get("task_type", "llm_call"), 0),
                    status_map.get(task["status"], 0),
                    unmet_deps,
                    is_act
                ]
        return obs

    def step(self, action):
        if action >= len(self.task_list):
            return self._get_obs(), -5.0, False, False, {"error": "Invalid empty task index"}
            
        selected_task = self.task_list[action]
        
        if selected_task["status"] == "completed":
            return self._get_obs(), -2.0, False, False, {"error": "Task already done"}
            
        unmet = self._get_unmet_deps(selected_task)
        if unmet > 0:
            return self._get_obs(), -5.0, False, False, {"error": f"Blocked by {unmet} dependencies"}
            
        # Execute Task successfully
        selected_task["status"] = "completed"
        reward = 10.0
        
        # Check if the whole board is complete
        terminated = all(t["status"] == "completed" for t in self.task_list)
        if terminated:
            reward += 100.0 # Massive reward for finishing the workflow
            
        return self._get_obs(), reward, terminated, False, {"success": f"Executed {selected_task['name']}"}