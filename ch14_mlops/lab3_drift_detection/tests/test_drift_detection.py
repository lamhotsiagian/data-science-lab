import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from drift_detection import *

def test_drift_detection():
    ref = generate_reference_data(200, 4)
    drifted = simulate_drift(ref, "mean_shift", 2.0)
    
    flags = detect_univariate_drift(ref, drifted)
    assert any(flags)
    
    flags_nodrift = detect_univariate_drift(ref, ref)
    assert not all(flags_nodrift)
    
    detector = AutoencoderDriftDetector()
    detector.fit(ref)
    ref_errors = detector.compute_reconstruction_error(ref)
    drift_errors = detector.compute_reconstruction_error(drifted)
    
    assert np.mean(drift_errors) > np.mean(ref_errors)
    
    sim_res = run_drift_simulation(n_features=4, drift_magnitudes=[0.0, 5.0])
    assert sim_res[5.0] > sim_res[0.0]
    
    assert plot_feature_distributions(ref, drifted) is not None
    assert plot_drift_scores([0.1, 0.9]) is not None
    assert plot_reconstruction_errors(ref_errors, drift_errors) is not None
    assert plot_drift_timeline([0.1, 0.2, 0.5], 0.3) is not None
