import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import pytest
from storytelling import *
import matplotlib.pyplot as plt

def test_climate():
    fig = story_climate_trends()
    assert len(fig.axes) == 1

def test_market():
    fig = story_market_comparison()
    assert len(fig.axes) == 1

def test_demographics():
    fig = story_demographic_analysis()
    assert len(fig.axes) == 1
