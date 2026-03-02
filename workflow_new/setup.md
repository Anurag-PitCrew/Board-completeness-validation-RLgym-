Step-by-Step Execution Flow

Open your terminal, ensure your virtual environment is activated, and navigate to the directory:
cd dev/workflow_new

Here is the exact order to run your pipeline to go from zero to a fully fine-tuned RL agent:

Phase 1: Data Generation

You need "expert" data before you can clone it.

Command: python generate_expert.py

What it does: Uses rule-based logic to perfectly solve the dependency graph in workflow_schema.json 100 times.

Output: Creates expert_workflow_data.npz.

(Alternative: If you have historical JSON data, run python convert_history_bc.py instead).

Phase 2: Behavior Cloning (Supervised Learning)

Train the neural network to mimic the expert data.

Command: python train_workflow_bc.py

What it does: Feeds expert_workflow_data.npz into a PyTorch Neural Network, minimizing the difference between the agent's guesses and the expert's actual moves.

Output: Creates workflow_bc.pth.

Phase 3: Verify the Clone

Make sure the BC agent actually learned the rules.

Command: python watch_workflow_bc.py

What it does: Loads workflow_bc.pth and runs a visual simulation in your terminal. You should see it cleanly executing tasks without triggering dependency errors.

Phase 4: Reinforcement Learning (Fine-Tuning)

Let the agent play the game to discover even better optimizations.

Command: python train_workflow_ppo.py

What it does: Uses Stable-Baselines3 PPO to let the agent explore the environment for 100,000 timesteps. It learns from the reward function (+10 for success, -5 for illegal moves).

Output: Creates workflow_ppo_final.zip and logs data to logs_workflow/.

Phase 5: Verify the RL Agent

Watch your final, fully-trained agent solve the board.

Command: python watch_workflow_ppo.py

What it does: Loads the PPO .zip model and renders the board. Because it was trained via RL, it should be highly resilient and optimally efficient.