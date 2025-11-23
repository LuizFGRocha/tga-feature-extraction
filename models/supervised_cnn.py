import torch
import torch.nn as nn
from models.base import TGAFeatureExtractor

class SupervisedCNN(TGAFeatureExtractor):
    def __init__(self, output_dim=25):
        """
        Args:
            output_dim: Number of regression targets. 
                        The AFM dataset Y has shape (5, 5) = 25 flattened features.
        """
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2), # Added dropout for small dataset
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.flat_dim = 128 * 256
        
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def encode(self, x):
        """Returns the features before the final regression layer."""
        x = self.conv_layers(x)
        return x

    def forward(self, x):
        features = self.conv_layers(x)
        prediction = self.regressor(features)
        return prediction