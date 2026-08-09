import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from svd_compression import generate_test_image, compress_svd, compute_psnr, compute_compression_ratio, compress_and_analyze

def test_compression():
    img = generate_test_image(32)
    rec = compress_svd(img, 5)
    assert rec.shape == (32, 32)
    
def test_psnr():
    img = np.ones((10,10))
    assert compute_psnr(img, img) == float('inf')
    
def test_compress_analyze():
    img = generate_test_image(32)
    res = compress_and_analyze(img, [1, 2])
    assert 1 in res
    assert 'psnr' in res[1]
