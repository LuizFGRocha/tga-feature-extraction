from models.attention_unet import AttentionUNet
from models.autoencoder import ConvAutoencoder
from models.variational_autoencoder import VAE
from models.bigger_variational_autoencoder import Bigger_VAE

def get_model(model_name, **kwargs):
    """
    Factory function to instantiate models.
    
    Args:
        model_name (str): Name of the model ('attention_unet', 'autoencoder', 'variational_autoencoder')
        **kwargs: Arguments passed to the model constructor
    """
    if model_name == 'attention_unet':
        return AttentionUNet(**kwargs)
    elif model_name == 'autoencoder':
        return ConvAutoencoder()
    elif model_name == 'variational_autoencoder':
        return VAE(**kwargs)
    elif model_name == 'bigger_variational_autoencoder':
        return Bigger_VAE(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: ['attention_unet', 'autoencoder', 'variational_autoencoder']")