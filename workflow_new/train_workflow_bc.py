import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class WorkflowBCAgent(nn.Module):
    # Updated default dimensions here just to be safe
    def __init__(self, input_dim=100, output_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def predict(self, obs):
        self.eval()
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            return torch.argmax(self.forward(t), dim=1).item()

if __name__ == "__main__":
    data = np.load("expert_workflow_data.npz")
    X = torch.FloatTensor(data["obs"])
    Y = torch.LongTensor(data["actions"])
    
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)
    
    # FIX: Explicitly set the new dimensions
    model = WorkflowBCAgent(input_dim=100, output_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    
    print("Training Behavior Clone on DAG Workflow...")
    for epoch in range(30):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "workflow_bc.pth")
    print("Saved workflow_bc.pth")