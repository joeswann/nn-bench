import torch
import torch.nn as nn
from enum import Enum

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LiquidTimeConstantNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, steps, step_size, solver='SemiImplicit', adaptive=True, use_embedding=True):
        super(LiquidTimeConstantNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.steps = steps
        self.step_size = step_size
        self.solver = solver
        self.adaptive = adaptive
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = nn.Embedding(input_size, hidden_size)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)

        self.recurrent_weights = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.time_constant = nn.Parameter(torch.empty(hidden_size))
        if adaptive:
            self.adaptive_weights = nn.Linear(hidden_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_embedding:
            nn.init.xavier_uniform_(self.embedding.weight)
        else:
            nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.recurrent_weights.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.uniform_(self.time_constant, 0.1, 1)  # Avoid very small time constants
        if self.adaptive:
            nn.init.xavier_uniform_(self.adaptive_weights.weight)

    def forward(self, input_sequence):
        batch_size, seq_length = input_sequence.size()[:2]
        
        if self.use_embedding:
            clipped_input = torch.clamp(input_sequence.long(), 0, self.input_size - 1)
            input_embeddings = self.embedding(clipped_input)
        else:
            input_embeddings = self.input_layer(input_sequence)

        hidden_states = []
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(input_sequence.device)

        for t in range(seq_length):
            input_t = input_embeddings[:, t]
            hidden_state = self.fused_solver(hidden_state, input_t)
            hidden_states.append(hidden_state)

        hidden_states = torch.stack(hidden_states, dim=1)
        output = self.output_layer(hidden_states)
        
        if torch.isnan(output).any():
            print(f"NaN detected in output. Hidden states: {hidden_states}")
        
        return output

    def ltc_ode(self, hidden_state, input_t):
        S = input_t + self.recurrent_weights(hidden_state)
        if self.adaptive:
            time_varying_constant = torch.clamp(self.time_constant * torch.tanh(S + self.adaptive_weights(hidden_state)), min=1e-6, max=1e6)
        else:
            time_varying_constant = torch.clamp(self.time_constant * torch.tanh(S), min=1e-6, max=1e6)
        d_hidden_state = (S - hidden_state) / time_varying_constant
        return d_hidden_state

    def fused_solver(self, hidden_state, input_t):
        input_t = input_t.float()  # Convert input_t to Float
        if self.solver == 'Explicit':
            return self._ode_step_explicit(hidden_state, input_t)
        elif self.solver == 'SemiImplicit':
            return self._ode_step(hidden_state, input_t)
        elif self.solver == 'RungeKutta':
            return self._ode_step_runge_kutta(hidden_state, input_t)
        else:
            raise ValueError(f"Unknown ODE solver '{self.solver}'")

    def _ode_step(self, hidden_state, input_t):
        for _ in range(self.steps):
            d_hidden_state = self.ltc_ode(hidden_state, input_t)
            hidden_state = hidden_state + self.step_size * torch.clamp(d_hidden_state, min=-1e6, max=1e6)
        return hidden_state

    def _ode_step_runge_kutta(self, hidden_state, input_t):
        for _ in range(self.steps):
            k1 = self.step_size * self.ltc_ode(hidden_state, input_t)
            k2 = self.step_size * self.ltc_ode(hidden_state + k1 * 0.5, input_t)
            k3 = self.step_size * self.ltc_ode(hidden_state + k2 * 0.5, input_t)
            k4 = self.step_size * self.ltc_ode(hidden_state + k3, input_t)
            delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            hidden_state = hidden_state + torch.clamp(delta, min=-1e6, max=1e6)
        return hidden_state

    def _ode_step_explicit(self, hidden_state, input_t):
        for _ in range(self.steps):
            d_hidden_state = self.ltc_ode(hidden_state, input_t)
            hidden_state = hidden_state + self.step_size * torch.clamp(d_hidden_state, min=-1e6, max=1e6)
        return hidden_state
