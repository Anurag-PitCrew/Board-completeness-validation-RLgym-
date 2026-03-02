import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit  # <-- NEW: Prevents infinite loops!
from kanban_env import KanbanEnv
import torch

def make_env():
    """Helper function to cleanly wrap the environment with limits and monitors."""
    env = KanbanEnv()
    # 1. Force the episode to end after 100 steps so training doesn't hang
    env = TimeLimit(env, max_episode_steps=100)
    # 2. Add the Monitor wrapper so SB3 can log stats without complaining
    env = Monitor(env)
    return env

def train_rl_agent():
    # Create securely wrapped environments
    env = make_env()
    eval_env = make_env()
    
    print("Checking environment compatibility...")
    check_env(env.unwrapped) # Check the base env to be safe
    print("Environment is valid! ✅\n")

    # The evaluation callback (Now safe from hanging)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/best_model/',
        log_path='./logs/results/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    device = torch.device("mps")
    print("Initializing PPO Agent on MPS...")
    model = PPO(
        "MlpPolicy", 
        env, 
        device=device, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log="./kanban_tensorboard/"
    )

    # Train the Agent
    timesteps = 100000
    print(f"Starting training for {timesteps} timesteps. This may take a few minutes...")
    
    model.learn(total_timesteps=timesteps, callback=eval_callback)

    model.save("kanban_ppo_final")
    print("Training complete and model saved as 'kanban_ppo_final.zip'!")

if __name__ == "__main__":
    os.makedirs("./logs/best_model/", exist_ok=True)
    os.makedirs("./logs/results/", exist_ok=True)
    
    train_rl_agent()