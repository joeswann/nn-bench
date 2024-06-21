import torch
import numpy as np
import matplotlib.pyplot as plt
from network_ltc import LiquidTimeConstantNetwork
from dataset_toy import ToyData

def load_model(model_path, input_size, hidden_size, output_size):
    model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps=5, step_size=0.1, solver='RungeKutta', adaptive=True, use_embedding=True)
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

if __name__ == "__main__":
    model_path = "model_ltc_toy.pt"
    dataset = ToyData()
    
    model = load_model(model_path, dataset.input_size, hidden_size=32, output_size=dataset.output_size)
    
    # Test the model's accuracy
    accuracy = test_accuracy(model, dataset.dataset)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Use a random sequence from the dataset as the initial sequence
    # initial_sequence = dataset.dataset.data[np.random.randint(len(dataset.dataset))][:50]
    
    # generated_sequence = generate_sequence(model, initial_sequence)
    
    # Plot the generated sequence
    # plt.figure(figsize=(12, 6))
    # plt.plot(generated_sequence)
    # plt.title('Generated Toy Sequence')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Class')
    # plt.savefig('toy_sequence.png')
    # plt.close()
    
    # print(f"Generated sequence plot saved as 'toy_sequence.png'")
