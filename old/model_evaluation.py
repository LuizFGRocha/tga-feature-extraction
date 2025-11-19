import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd

from models.attention_unet import AttentionUNet
from models.autoencoder import ConvAutoencoder

def load_model(model_name, checkpoint_path, device):
    if model_name == 'attention_unet':
        model = AttentionUNet(ch_in=2, ch_out=2, compressed_dim=64)
        # The AttentionUNet checkpoint contains more than just the state_dict
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == 'autoencoder':
        model = ConvAutoencoder()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    model.double()
    model.eval()
    return model

def generate_encodings(model, data_loader, device):
    encodings = []
    with torch.no_grad():
        for x_batch, in data_loader:
            x_batch = x_batch.to(device)
            encoding = model.encode(x_batch).cpu().numpy()
            encodings.append(encoding)
    return np.vstack(encodings)

def evaluate_with_bootstrap(encodings, labels, n_bootstraps=1000, alpha=0.05):
    """
    Evaluates encoding quality using bootstrapping
    """
    n_samples = len(encodings)
    bootstrap_scores = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = encodings[indices], labels[indices]
        
        oob_indices = np.array(list(set(range(n_samples)) - set(indices)))
        if len(oob_indices) == 0:
            continue 
            
        X_oob, y_oob = encodings[oob_indices], labels[oob_indices]

        regressor = Ridge()
        regressor.fit(X_boot, y_boot)
    
        score = regressor.score(X_oob, y_oob)
        bootstrap_scores.append(score)
        
    lower_bound = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, lower_bound, upper_bound

def evaluate_with_cv(encodings, labels, n_splits=5, n_repeats=20, alpha=0.05):
    """
    Evaluates encoding quality using Repeated K-Fold Cross-Validation
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    cv_scores = []

    for train_idx, test_idx in cv.split(encodings):
        X_train, X_test = encodings[train_idx], encodings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        regressor = Ridge()
        regressor.fit(X_train, y_train)

        score = regressor.score(X_test, y_test)
        cv_scores.append(score)

    mean_score = np.mean(cv_scores)
    lower_bound = np.percentile(cv_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(cv_scores, 100 * (1 - alpha / 2))

    return mean_score, lower_bound, upper_bound

def main():
    parser = argparse.ArgumentParser(description="Evaluate encoding quality of models.")
    parser.add_argument('--model_name', type=str, required=True, choices=['attention_unet', 'autoencoder'], help='Name of the model to evaluate.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--data_path', type=str, default='./data/tga_afm/data.npz', help='Path to the dataset.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    data = np.load(args.data_path)
    X = data['X'][:, 1:3, :]  # Use W and dW/dT curves
    Y = data['Y']
    
    dataset = TensorDataset(torch.from_numpy(X))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = load_model(args.model_name, args.checkpoint_path, device)
    encodings = generate_encodings(model, data_loader, device)
    
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)

    # target labels from the dataset
    label_configs = [
        {'name': 'Height Mean',       'data': Y[:, 2, 0]},
        {'name': 'Min Ferret Skewness', 'data': Y[:, 0, 2]},
        {'name': 'Area Mean',         'data': Y[:, 3, 0]},
        {'name': 'Volume Mean',       'data': Y[:, 4, 0]},
    ]
    
    results = []
    print("\n--- Encoding Quality Evaluation ---")
    for config in label_configs:
        labels = config['data']
        mean_r2, lower_ci, upper_ci = evaluate_with_cv(encodings_scaled, labels)
        results.append({
            'Target Property': config['name'],
            'Mean R^2': mean_r2,
            '95% CI Lower': lower_ci,
            '95% CI Upper': upper_ci
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()