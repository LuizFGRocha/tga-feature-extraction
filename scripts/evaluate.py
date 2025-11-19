import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Add project root to path so we can import from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the new modular structure
from models.factory import get_model
from src.dataset import TGADataset
from src.evaluation import evaluate_with_bootstrap, evaluate_with_cv

def generate_encodings(model, data_loader, device):
    """
    Runs inference to generate encodings and collects corresponding labels.
    """
    encodings = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            
            # Generate encoding
            # Note: Assuming models inherit from TGAFeatureExtractor and have .encode()
            encoding = model.encode(x).cpu().numpy()
            
            encodings.append(encoding)
            all_labels.append(y.numpy())
            
    return np.vstack(encodings), np.vstack(all_labels)

def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on device: {device}")
    # Load Data
    # We use mode='feature' to get (X, Y) pairs from the dataset
    dataset = TGADataset(data_path=args.data_path, mode='feature')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Load Model
    print(f"Loading {args.model_name} from {args.checkpoint_path}...")
    model = get_model(args.model_name, compressed_dim=args.latent_dim)
    
    # Use the standardized load_checkpoint method from the Base class
    # If your models don't inherit from Base yet, you might need manual loading here
    try:
        model.load_checkpoint(args.checkpoint_path, device)
    except AttributeError:
        # Fallback for legacy models not yet inheriting from Base
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    model.to(device)
    model.double() # Ensure precision matches data

    # Generate Encodings
    print("Generating encodings...")
    encodings, Y = generate_encodings(model, data_loader, device)
    
    # Scale encodings (important for Ridge regression)
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)

    # Define Targets
    # Y shape is (Samples, Characteristics, Statistics)
    # 0=Min Ferret, 2=Height, 3=Area, 4=Volume
    # Statistic 0=Mean, 2=Skewness
    label_configs = [
        {'name': 'Height Mean',       'data': Y[:, 2, 0]},
        {'name': 'Min Ferret Skewness', 'data': Y[:, 0, 2]},
        {'name': 'Area Mean',         'data': Y[:, 3, 0]},
        {'name': 'Volume Mean',       'data': Y[:, 4, 0]},
    ]
    
    # Run Evaluation
    results = []
    print(f"\n--- Encoding Quality Evaluation ({args.method}) ---")
    
    for config in label_configs:
        labels = config['data']
        
        if args.method == 'bootstrap':
            mean_r2, lower_ci, upper_ci = evaluate_with_bootstrap(encodings_scaled, labels)
        else:
            mean_r2, lower_ci, upper_ci = evaluate_with_cv(encodings_scaled, labels)
            
        results.append({
            'Target Property': config['name'],
            'Mean R^2': mean_r2,
            '95% CI Lower': lower_ci,
            '95% CI Upper': upper_ci
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate encoding quality of TGA models.")
    parser.add_argument('--model_name', type=str, required=True, choices=['attention_unet', 'autoencoder'], help='Model architecture name.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the .pth checkpoint.')
    parser.add_argument('--data_path', type=str, default='./data/tga_afm/data.npz', help='Path to the evaluation dataset (must contain X and Y).')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of the latent space.')
    parser.add_argument('--method', type=str, default='bootstrap', choices=['bootstrap', 'cv'], help='Evaluation method.')
    args = parser.parse_args()

    results = run_evaluation(args)
    print(pd.DataFrame(results))
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()