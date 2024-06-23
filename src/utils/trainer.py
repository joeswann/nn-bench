import torch
import numpy as np

class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, device, num_epochs, gradient_clip):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip

    def train(self):
        self.model.to(self.device)
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in self.dataloader:
                input_data = batch["input"].to(self.device)
                target_data = batch["target"].to(self.device)

                # Forward pass
                outputs = self.model(input_data)

                # Print shapes for debugging
                print(f"Input shape: {input_data.shape}")
                print(f"Target shape: {target_data.shape}")
                print(f"Output shape: {outputs.shape}")

                # Compute the loss
                loss = self.criterion(outputs, target_data)

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected. Input: {input_data}, Target: {target_data}")
                    print(f"Model output: {outputs}")
                    continue

                # Print shapes for debugging
                print(f"Input shape: {input_data.shape}")
                print(f"Target shape: {target_data.shape}")
                print(f"Output shape: {outputs.shape}")

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                # Check for NaN gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradient detected in {name}")
                            param.grad = torch.where(torch.isnan(param.grad), torch.zeros_like(param.grad), param.grad)

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

            # Early stopping if loss is NaN
            if np.isnan(avg_loss):
                print("Training stopped due to NaN loss")
                break
