import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentMoE(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, num_layers, k=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.k = k 
        self.projectors = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_layers)
        ])
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.routers = nn.ModuleList([
            nn.Linear(hidden_size, num_experts) for _ in range(num_layers)
        ])
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size, 4 * input_size),
                    nn.ReLU(),
                    nn.Linear(4 * input_size, input_size)
                ) for _ in range(num_experts)
            ]) for _ in range(num_layers)
        ])