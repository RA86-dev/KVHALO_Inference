import torch

def simulate_2bit_quantization(tensor):
    """
    Simulates the heavy information loss of 2-bit quantization for training.
    We convert the continuous vectors into a 4-step staircase to train Candy to denoise it.
    """
    min_val, max_val = tensor.min(), tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val + 1e-5)

    # 2-bit allows exactly 4 values (0, 1, 2, 3)
    quantized_2bit = torch.round(normalized * 3.0) / 3.0

    # Scale back up to original magnitude space
    return (quantized_2bit * (max_val - min_val)) + min_val