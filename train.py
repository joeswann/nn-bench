import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from enum import Enum

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LiquidTimeConstantNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, steps, step_size, solver=ODESolver.SemiImplicit, adaptive=True):
        super(LiquidTimeConstantNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.steps = steps
        self.step_size = step_size
        self.solver = solver
        self.adaptive = adaptive
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        self.weights = nn.Linear(input_size, hidden_size)
        self.recurrent_weights = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.time_constant = nn.Parameter(torch.empty(hidden_size))
        if adaptive:
            self.adaptive_weights = nn.Linear(hidden_size, hidden_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.weight)
        nn.init.xavier_uniform_(self.recurrent_weights.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.uniform_(self.time_constant, -1, 1)
        if self.adaptive:
            nn.init.xavier_uniform_(self.adaptive_weights.weight)
        
    def forward(self, input_sequence):
        batch_size, seq_length = input_sequence.size()
        input_embeddings = self.embedding(input_sequence)

        hidden_states = []
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(input_sequence.device)
        
        for t in range(seq_length):
            input_t = input_embeddings[:, t]
            hidden_state = self.fused_solver(hidden_state, input_t)
            hidden_states.append(hidden_state)
        
        hidden_states = torch.stack(hidden_states, dim=1)
        output = self.output_layer(hidden_states)
        return output
    
    def ltc_ode(self, hidden_state, input_t):
        S = input_t + self.recurrent_weights(hidden_state)
        if self.adaptive:
            time_varying_constant = self.time_constant * torch.tanh(S + self.adaptive_weights(hidden_state))
        else:
            time_varying_constant = self.time_constant * torch.tanh(S)
        d_hidden_state = (S - hidden_state) / time_varying_constant
        return d_hidden_state
    
    def fused_solver(self, hidden_state, input_t):
        input_t = input_t.float()  # Convert input_t to Float
        if self.solver == ODESolver.Explicit:
            return self._ode_step_explicit(hidden_state, input_t)
        elif self.solver == ODESolver.SemiImplicit:
            return self._ode_step(hidden_state, input_t)
        elif self.solver == ODESolver.RungeKutta:
            return self._ode_step_runge_kutta(hidden_state, input_t)
        else:
            raise ValueError(f"Unknown ODE solver '{self.solver}'")
    
    def _ode_step(self, hidden_state, input_t):
        for _ in range(self.steps):
            d_hidden_state = self.ltc_ode(hidden_state, input_t.float())
            hidden_state = hidden_state + self.step_size * d_hidden_state
        return hidden_state
    
    def _ode_step_runge_kutta(self, hidden_state, input_t):
        for _ in range(self.steps):
            k1 = self.step_size * self.ltc_ode(hidden_state, input_t)
            k2 = self.step_size * self.ltc_ode(hidden_state + k1 * 0.5, input_t)
            k3 = self.step_size * self.ltc_ode(hidden_state + k2 * 0.5, input_t)
            k4 = self.step_size * self.ltc_ode(hidden_state + k3, input_t)
            hidden_state = hidden_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return hidden_state
    
    def _ode_step_explicit(self, hidden_state, input_t):
        for _ in range(self.steps):
            d_hidden_state = self.ltc_ode(hidden_state, input_t)
            hidden_state = hidden_state + self.step_size * d_hidden_state
        return hidden_state

def train_ltc(model, inputs, targets, epochs, learning_rate, gradient_clip=1.0, save_path='ltc_model.pt'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for input_seq, target_seq in zip(inputs, targets):
            optimizer.zero_grad()
            output_seq = model(input_seq)
            loss = criterion(output_seq, target_seq)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(inputs)}')
    
    torch.save(model.state_dict(), save_path)

# Example usage
hidden_size = 32
steps = 5
step_size = 0.01
learning_rate = 0.001
num_epochs = 10
epochs = 50
gradient_clip = 1.0
save_path = 'ltc_model.pt'


# # Generate random input and target sequences
# batch_size = 32
# seq_length = 20
# inputs = [torch.randn(batch_size, seq_length, input_size) for _ in range(100)]
# targets = [torch.randn(batch_size, output_size) for _ in range(100)]
#
# train_ltc(model, inputs, targets, epochs, learning_rate, gradient_clip, save_path)




# Load the dataset and tokenize
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])

# Convert dataset to PyTorch DataLoader
tokenized_dataset.set_format(type='torch', columns=['input_ids'])
dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# Create the model
input_size = len(tokenizer.vocab)
output_size = len(tokenizer.vocab)
hidden_size = 32
steps = 5
step_size = 0.01
learning_rate = 0.001
num_epochs = 10
gradient_clip = 1.0
save_path = 'ltc_model.pt'
model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps, step_size, solver=ODESolver.SemiImplicit, adaptive=True)


# Load the dataset and tokenize
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])

# Convert dataset to PyTorch DataLoader
tokenized_dataset.set_format(type='torch', columns=['input_ids'])
dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# Create the model
input_size = len(tokenizer.vocab)
output_size = len(tokenizer.vocab)
hidden_size = 32
steps = 5
step_size = 0.01
learning_rate = 0.001
num_epochs = 10
gradient_clip = 1.0
save_path = 'ltc_model.pt'
model = LiquidTimeConstantNetwork(input_size, hidden_size, output_size, steps, step_size, solver=ODESolver.SemiImplicit, adaptive=True)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].long()
        
        # Forward pass
        outputs = model(input_ids)
        
        # Reshape outputs to (batch_size * seq_length, output_size)
        outputs = outputs.view(-1, output_size)
        # Reshape input_ids to (batch_size * seq_length)
        input_ids = input_ids.view(-1)
        
        if outputs.size(0) != input_ids.size(0):
            raise ValueError(f"Mismatch in output and input size: {outputs.size(0)} vs {input_ids.size(0)}")
        
        # Compute the loss
        loss = criterion(outputs, input_ids)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Save the final model weights
torch.save(model.state_dict(), save_path)

# Generate text using the trained model
generated_text = []
seed_text = "The quick brown fox"
input_sequence = tokenizer.encode(seed_text, return_tensors="pt")

for _ in range(100):  # Generate 100 words
    outputs = model(input_sequence)
    word_probs = torch.softmax(outputs[:, -1, :], dim=1)
    word_idx = torch.multinomial(word_probs, num_samples=1).item()
    input_sequence = torch.cat([input_sequence, torch.tensor([[word_idx]])], dim=1)
    generated_text.append(word_idx)

generated_text = tokenizer.decode(generated_text)
print("Generated Text:")
print(generated_text)
