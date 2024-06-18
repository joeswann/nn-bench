import torch
from train import LiquidTimeConstantNetwork, ODESolver

# Load the saved model
input_size = 10
hidden_size = 32
output_size = 5
steps = 5
step_size = 0.01
save_path = 'ltc_model.pt'

model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps, step_size, solver=ODESolver.SemiImplicit, adaptive=True)
model.load_state_dict(torch.load(save_path))
model.eval()

# Example usage of the loaded model
input_seq = torch.randn(1, 20, input_size)  # Adjust the input size as needed
with torch.no_grad():
    output = model(input_seq)
    print('Model output:', output)
