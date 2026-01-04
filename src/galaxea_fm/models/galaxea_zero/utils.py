import torch
from transformers import AutoImageProcessor
from PIL import Image


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch,
        num_key_value_heads * n_rep,
        slen,
        head_dim,
    )


class ImageProcessorToTransform:
    def __init__(self, processor: AutoImageProcessor):
        """
        Initialize the wrapper with an ImageProcessor instance.

        Args:
            processor (AutoImageProcessor): The Hugging Face ImageProcessor instance
        """
        self.processor = processor

    def __call__(self, img: Image, **kwargs: str) -> torch.Tensor:
        """
        Process the input image and return a PyTorch tensor.
        
        Args:
            img (PIL.Image): The input image to process.
        
        Returns:
            torch.Tensor: Processed image as a tensor ready for model input.
        """
        # Process the image using the ImageProcessor
        inputs = self.processor(img, return_tensors="pt", **kwargs)
        
        # Return the 'pixel_values' which is the processed tensor
        return inputs['pixel_values']
