import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.ltc_network import LiquidTimeConstantNetwork
from models.gru_network import GRUNetwork
from models.lstm_network import LSTMNetwork
from data.toy_dataset import ToyData

def load_model(model_type, input_size, hidden_size, output_size):
    model_path = f"models/model_{model_type}_toy.pt"
    
    if model_type == 'ltc':
        model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps=5, step_size=0.1, solver='RungeKutta', adaptive=True, use_embedding=True)
    elif model_type == 'gru':
        model = GRUNetwork(input_size, hidden_size, output_size, num_layers=1, use_embedding=True)
    elif model_type == 'lstm':
        model = LSTMNetwork(input_size, hidden_size, output_size, num_layers=1, use_embedding=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_sequence(model, initial_sequence, num_steps=100):
    input_sequence = torch.tensor(initial_sequence, dtype=torch.long).unsqueeze(0)
    generated_sequence = input_sequence.squeeze().tolist()

    for _ in range(num_steps):
        with torch.no_grad():
            output = model(input_sequence)
        next_class = torch.argmax(output[0, -1]).item()
        generated_sequence.append(next_class)
        input_sequence = torch.cat([input_sequence[:, 1:], torch.tensor([[next_class]])], dim=1)

    return generated_sequence

def test_accuracy(model, dataset):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataset:
            input_sequence = data['input'].unsqueeze(0)
            target_sequence = data['target']

            output = model(input_sequence)
            predicted_classes = torch.argmax(output[0], dim=-1)

            correct += (predicted_classes == target_sequence).sum().item()
            total += len(target_sequence)

    accuracy = correct / total
    return accuracy

def main(args):
    dataset = ToyData()
    
    model = load_model(args.model_type, dataset.input_size, args.hidden_size, dataset.output_size)
    
    # Test the model's accuracy
    accuracy = test_accuracy(model, dataset.dataset)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Generate a sequence
    initial_sequence = dataset.dataset.data[np.random.randint(len(dataset.dataset))][:args.sequence_length]
    generated_sequence = generate_sequence(model, initial_sequence, args.num_steps)
    
    # Plot the generated sequence
    plt.figure(figsize=(12, 6))
    plt.plot(generated_sequence)
    plt.title('Generated Toy Sequence')
    plt.xlabel('Time Steps')
    plt.ylabel('Class')
    plt.savefig('toy_sequence.png')
    plt.close()
    
    print(f"Generated sequence plot saved as 'toy_sequence.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy Data Inference")
    parser.add_argument("--model_type", choices=['ltc', 'gru', 'lstm'], required=True, help="Type of the model")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of the network")
    parser.add_argument("--sequence_length", type=int, default=50, help="Length of input sequence")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to generate")
    args = parser.parse_args()
    
    main(args)
