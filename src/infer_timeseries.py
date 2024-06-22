import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ltc_network import LiquidTimeConstantNetwork
from models.gru_network import GRUNetwork
from models.lstm_network import LSTMNetwork
from datasets.timeseries_dataset import TimeSeriesData

def load_model(model_type, input_size, hidden_size, output_size):
    model_path = f"models/model_{model_type}_timeseries.pt"
    
    if model_type == 'ltc':
        model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps=10, step_size=0.01, solver='RungeKutta', adaptive=True, use_embedding=False)
    elif model_type == 'gru':
        model = GRUNetwork(input_size, hidden_size, output_size, num_layers=1, use_embedding=False)
    elif model_type == 'lstm':
        model = LSTMNetwork(input_size, hidden_size, output_size, num_layers=1, use_embedding=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_next_steps(model, initial_sequence, num_steps=50):
    input_sequence = torch.tensor(initial_sequence, dtype=torch.float32).unsqueeze(0)
    predictions = []

    for _ in range(num_steps):
        with torch.no_grad():
            output = model(input_sequence)
        next_step = output[0, -1, :].numpy()
        predictions.append(next_step)
        input_sequence = torch.cat([input_sequence[:, 1:], torch.tensor(next_step).unsqueeze(0).unsqueeze(0)], dim=1)

    return np.array(predictions)

def main(args):
    dataset = TimeSeriesData()
    
    model = load_model(args.model_type, dataset.input_size, args.hidden_size, dataset.output_size)
    
    initial_sequence = dataset.data[-args.sequence_length:]
    
    predictions = predict_next_steps(model, initial_sequence, args.num_steps)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Inference")
    parser.add_argument("--model_type", choices=['ltc', 'gru', 'lstm'], required=True, help="Type of the model")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the network")
    parser.add_argument("--sequence_length", type=int, default=50, help="Length of input sequence")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of steps to predict")
    args = parser.parse_args()
    
    main(args)
