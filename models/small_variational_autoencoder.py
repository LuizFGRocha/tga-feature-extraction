import torch
import torch.nn as nn

from models.base import TGAFeatureExtractor

class SmallVAE(TGAFeatureExtractor):
    def __init__(self, compressed_dim=16, dropout_prob=0.2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Flatten()
        )
        
        flatten_dim = 128 * 64
        
        self.fc_mu = nn.Linear(flatten_dim, compressed_dim)
        self.fc_var = nn.Linear(flatten_dim, compressed_dim)
        
        self.decoder_input = nn.Linear(compressed_dim, flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(16, 2, 4, stride=2, padding=1),
            
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x):
        x = self.conv_layers(x)
        return self.fc_mu(x)

    def forward(self, x):
        flat = self.conv_layers(x)
        mu = self.fc_mu(flat)
        logvar = self.fc_var(flat)
        
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_input(z)
        d = d.view(-1, 128, 64)
        reconstruction = self.decoder(d)
        
        return reconstruction, mu, logvar