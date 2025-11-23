import torch
import torch.nn as nn

# Assuming models.base is available in your environment
from models.base import TGAFeatureExtractor

class NanoVAE(TGAFeatureExtractor):
    def __init__(self, compressed_dim=8, dropout_prob=0.1): # Reduced dim to 8 due to small data
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Input: (N, 2, 1024)
            # L1: 1024 -> 512
            nn.Conv1d(2, 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            
            # L2: 512 -> 256
            nn.Conv1d(8, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            # L3: 256 -> 128
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            # L4: 128 -> 64 (Added layer to reduce spatial dim further)
            nn.Conv1d(32, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
        )
        
        self.flatten_dim = 32 * 64 
        
        self.fc_mu = nn.Linear(self.flatten_dim, compressed_dim)
        self.fc_var = nn.Linear(self.flatten_dim, compressed_dim)
        
        self.decoder_input = nn.Linear(compressed_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            # Unflatten: (N, 32, 64) -> (N, 32, 128)
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            # (N, 32, 128) -> (N, 16, 256)
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
             
            # (N, 16, 256) -> (N, 8, 512)
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            
            # (N, 8, 512) -> (N, 2, 1024)
            nn.ConvTranspose1d(8, 2, 4, stride=2, padding=1)
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
        x = x.view(x.size(0), -1) # Flatten
        return self.fc_mu(x)

    def forward(self, x):
        out = self.conv_layers(x)
        flat = out.view(out.size(0), -1) # Flatten spatially
        
        mu = self.fc_mu(flat)
        logvar = self.fc_var(flat)
        
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_input(z)
        d = d.view(-1, 32, 64) # Reshape back to tensor
        
        reconstruction = self.decoder(d)
        
        return reconstruction, mu, logvar