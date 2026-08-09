import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_test_image(size=256):
    img = np.zeros((size, size))
    img[size//4:size//2, size//4:size//2] = 1.0
    img[size//2:3*size//4, size//2:3*size//4] = 0.5
    return img

def compress_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return reconstructed

def compute_compression_ratio(original, k):
    m, n = original.shape
    original_size = m * n
    compressed_size = k * (m + n + 1)
    return original_size / compressed_size

def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 1.0 if original.max() <= 1.0 else 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compress_and_analyze(image, k_values):
    results = {}
    for k in k_values:
        rec = compress_svd(image, k)
        results[k] = {
            'psnr': compute_psnr(image, rec),
            'ratio': compute_compression_ratio(image, k)
        }
    return results

def plot_compression_comparison(image, k_values):
    pass # Visualizer stub
    
def plot_quality_vs_compression(results):
    pass # Visualizer stub
