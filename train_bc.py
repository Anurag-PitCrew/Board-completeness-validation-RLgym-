import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from kanban_env import KanbanEnv  # Import your environment

# ==========================================
# 1. Define the Neural Network (The "Brain")
# ==========================================
class KanbanBCAgent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KanbanBCAgent, self).__init__()
        # We flatten the (15, 5) Kanban board into a 75-element vector
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim) # Outputs probabilities for each task index
        )

    def forward(self, x):
        # Flatten the 2D observation matrix into a 1D vector
        x = x.view(x.size(0), -1) 
        return self.network(x)

    def predict(self, obs):
        """Helper to use the model during testing."""
        self.eval()
        with torch.no_grad():
            # Convert NumPy array to PyTorch tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) 
            logits = self.forward(obs_tensor)
            # Pick the action with the highest probability
            action = torch.argmax(logits, dim=1).item()
        return action

# ==========================================
# 2. Load Data & Train
# ==========================================
def train_behavior_clone():
    # Load the expert dataset
    print("Loading expert data...")
    dataset = np.load("expert_kanban_data.npz")
    observations = dataset["obs"]
    actions = dataset["actions"]
    
    print(f"Loaded {len(observations)} state-action pairs.")

    # Convert to PyTorch Tensors
    X_tensor = torch.FloatTensor(observations)
    y_tensor = torch.LongTensor(actions)

    # Create a DataLoader for batching
    train_data = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    input_dim = 15 * 5  # max_tasks (15) * features (5)
    output_dim = 15     # Action space size (Discrete 15)
    
    model = KanbanBCAgent(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 50
    print("\nStarting Training...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X)
            
            # Calculate loss (how far off is the model from the expert?)
            loss = criterion(predictions, batch_y)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    print("Training complete!")
    return model

# ==========================================
# 3. Test the Trained Agent
# ==========================================
def test_agent(model, episodes=3):
    env = KanbanEnv()
    print("\nTesting the trained Behavior Clone...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            # The agent predicts the best action based on the observation
            action = model.predict(obs)
            
            obs, reward, terminated, _, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or step_count > 50: # Cap at 50 steps for testing
                break
                
        print(f"Test Episode {ep + 1} | Steps: {step_count} | Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # 1. Train the model
    trained_model = train_behavior_clone()
    
    # 2. Save the model weights
    torch.save(trained_model.state_dict(), "kanban_bc_model.pth")
    print("Model saved to 'kanban_bc_model.pth'")
    
    # 3. Test it in the environment
    test_agent(trained_model)