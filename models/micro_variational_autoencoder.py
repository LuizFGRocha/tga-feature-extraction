import torch
import torch.nn as nn

from models.base import TGAFeatureExtractor

class MicroVAE(TGAFeatureExtractor):
    def __init__(self, compressed_dim=8, dropout_prob=0.1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            nn.AdaptiveAvgPool1d(1),
            
            nn.Flatten()
        )
        
        flatten_dim = 64
        
        self.fc_mu = nn.Linear(flatten_dim, compressed_dim)
        self.fc_var = nn.Linear(flatten_dim, compressed_dim)
        
        self.decoder_input = nn.Linear(compressed_dim, 64 * 128)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
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
        d = d.view(-1, 64, 128) 
        reconstruction = self.decoder(d)
        
        return reconstruction, mu, logvar