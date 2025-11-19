from models.attention_unet import AttentionUNet
from models.autoencoder import ConvAutoencoder

def get_model(model_name, **kwargs):
    """
    Factory function to instantiate models.
    
    Args:
        model_name (str): Name of the model ('attention_unet', 'autoencoder')
        **kwargs: Arguments passed to the model constructor
    """
    if model_name == 'attention_unet':
        return AttentionUNet(**kwargs)
    elif model_name == 'autoencoder':
        return ConvAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available: ['attention_unet', 'autoencoder']")