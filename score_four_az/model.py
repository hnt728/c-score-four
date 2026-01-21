import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
