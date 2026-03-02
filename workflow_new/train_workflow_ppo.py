import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from workflow_env import WorkflowEnv
import torch

def make_env():
    """Cleanly wrap the environment to prevent infinite loops during RL training."""
    env = WorkflowEnv()
    # Force episode to end after 100 steps to prevent the agent from getting stuck
    env = TimeLimit(env, max_episode_steps=100)
    # Monitor wrapper for Stable-Baselines3 logging
    env = Monitor(env)
    return env

def train_rl_workflow_agent():
    env = make_env()
    eval_env = make_env()
    
    print("Checking Workflow Environment compatibility...")
    check_env(env.unwrapped) 
    print("Environment is valid! ✅\n")

    # Save the best performing model during training
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs_workflow/best_model/',
        log_path='./logs_workflow/results/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    # Auto-detect hardware acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    #if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing PPO Agent on {device}...")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        device=device, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log="./workflow_tensorboard/"
    )

    # Train the Agent (Adjust timesteps as needed)
    timesteps = 100000
    print(f"Starting PPO training for {timesteps} timesteps. Let the agent explore the DAG...")
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    model.save("workflow_ppo_final")
    print("Training complete! Model saved as 'workflow_ppo_final.zip'")

if __name__ == "__main__":
    os.makedirs("./logs_workflow/best_model/", exist_ok=True)
    os.makedirs("./logs_workflow/results/", exist_ok=True)
    
    train_rl_workflow_agent()