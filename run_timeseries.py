import torch
import numpy as np
import matplotlib.pyplot as plt
from network_ltc import LiquidTimeConstantNetwork
from dataset_timeseries import TimeSeriesData

def load_model(model_path, input_size, hidden_size, output_size):
    model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps=10, step_size=0.01, solver='RungeKutta', adaptive=True, use_embedding=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_next_steps(model, initial_sequence, num_steps=50):
    input_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0)
    predictions = []

    for _ in range(num_steps):
        with torch.no_grad():
            output = model(input_sequence)
        next_step = output[0, -1, :20].numpy()  # Only use the stock returns, not sentiment
        predictions.append(next_step)
        input_sequence = torch.cat([input_sequence[:, 1:], torch.tensor(next_step).unsqueeze(0).unsqueeze(0)], dim=1)

    return np.array(predictions)

if __name__ == "__main__":
    model_path = "models/ltc_model_timeseries.pt"
    dataset = TimeSeriesData()
    
    model = load_model(model_path, dataset.input_size, hidden_size=32, output_size=dataset.output_size)
    
    # Use the last sequence from the dataset as the initial sequence
    initial_sequence = dataset.data[-1]
    
    predictions = predict_next_steps(model, initial_sequence)
    
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    for i in range(predictions.shape[1]):
        plt.plot(predictions[:, i], label=f'Stock {i+1}')
    plt.title('Predicted Stock Returns')
    plt.xlabel('Time Steps')
    plt.ylabel('Returns')
    plt.legend()
    plt.savefig('timeseries_predictions.png')
    plt.close()
    
    print(f"Predictions plot saved as 'timeseries_predictions.png'")
