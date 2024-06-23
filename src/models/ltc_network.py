import torch
import torch.nn as nn
from enum import Enum
import math

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LiquidTimeConstantNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, steps=10, step_size=0.01, solver='RungeKutta', adaptive=True, use_embedding=False):
        super(LiquidTimeConstantNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.steps = steps
        self.step_size = step_size
        self.solver = ODESolver[solver]
        self.adaptive = adaptive
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)

        self.recurrent_weights = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.time_constant = nn.Parameter(torch.empty(hidden_size))
        if adaptive:
            self.adaptive_weights = nn.Linear(hidden_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_embedding:
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            nn.init.xavier_uniform_(self.input_layer.weight)
            nn.init.zeros_(self.input_layer.bias)
        
        nn.init.orthogonal_(self.recurrent_weights.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        nn.init.uniform_(self.time_constant, 0.1, 1.0)
        
        if self.adaptive:
            nn.init.xavier_uniform_(self.adaptive_weights.weight)

    def forward(self, input_sequence):
        batch_size, seq_length, input_dim = input_sequence.size()
        
        if self.use_embedding:
            input_embeddings = self.embedding(input_sequence.long())
        else:
            input_embeddings = self.input_layer(input_sequence)

        hidden_state = torch.zeros(batch_size, self.hidden_size, device=input_sequence.device)
        outputs = []

        for t in range(seq_length):
            input_t = input_embeddings[:, t]
            hidden_state = self.fused_solver(hidden_state, input_t)
            output = self.output_layer(hidden_state)
            outputs.append(output.unsqueeze(1))

        # Concatenate outputs along the second dimension (seq_length)
        outputs = torch.cat(outputs, dim=1)
        
        # Ensure the output shape matches the input shape
        outputs = outputs.view(batch_size, seq_length, self.output_size)

        return outputs

    def ltc_ode(self, hidden_state, input_t):
        S = input_t + self.recurrent_weights(hidden_state)
        if self.adaptive:
            time_varying_constant = self.time_constant * torch.sigmoid(S + self.adaptive_weights(hidden_state))
        else:
            time_varying_constant = self.time_constant * torch.sigmoid(S)
        time_varying_constant = torch.clamp(time_varying_constant, min=1e-6, max=1e6)
        d_hidden_state = (S - hidden_state) / time_varying_constant
        return d_hidden_state

    def fused_solver(self, hidden_state, input_t):
        if self.solver == ODESolver.Explicit:
            return self._ode_step_explicit(hidden_state, input_t)
        elif self.solver == ODESolver.SemiImplicit:
            return self._ode_step_semi_implicit(hidden_state, input_t)
        elif self.solver == ODESolver.RungeKutta:
            return self._ode_step_runge_kutta(hidden_state, input_t)
        else:
            raise ValueError(f"Unknown ODE solver '{self.solver}'")

    def _ode_step_semi_implicit(self, hidden_state, input_t):
        for _ in range(self.steps):
            d_hidden_state = self.ltc_ode(hidden_state, input_t)
            hidden_state = hidden_state + self.step_size * torch.clamp(d_hidden_state, min=-1e6, max=1e6)
        return hidden_state

    def _ode_step_runge_kutta(self, hidden_state, input_t):
        for _ in range(self.steps):
            k1 = self.ltc_ode(hidden_state, input_t)
            k2 = self.ltc_ode(hidden_state + 0.5 * self.step_size * k1, input_t)
            k3 = self.ltc_ode(hidden_state + 0.5 * self.step_size * k2, input_t)
            k4 = self.ltc_ode(hidden_state + self.step_size * k3, input_t)
            
            delta = (k1 + 2 * k2 + 2 * k3 + k4) * (self.step_size / 6)
            hidden_state = hidden_state + torch.clamp(delta, min=-1e6, max=1e6)
        return hidden_state

    def _ode_step_explicit(self, hidden_state, input_t):
        for _ in range(self.steps):
            d_hidden_state = self.ltc_ode(hidden_state, input_t)
            hidden_state = hidden_state + self.step_size * torch.clamp(d_hidden_state, min=-1e6, max=1e6)
        return hidden_state

    def extra_repr(self):
        return (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
                f'output_size={self.output_size}, steps={self.steps}, '
                f'step_size={self.step_size}, solver={self.solver.name}, '
                f'adaptive={self.adaptive}, use_embedding={self.use_embedding}')
