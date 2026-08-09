import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

def story_climate_trends():
    fig, ax = plt.subplots()
    x = np.arange(1900, 2020)
    y = np.linspace(14, 15, len(x)) + np.random.normal(0, 0.1, len(x))
    ax.plot(x, y)
    ax.set_title("Climate Trends")
    return fig

def story_market_comparison():
    fig, ax = plt.subplots()
    x = np.arange(2010, 2020)
    y = np.random.dirichlet(np.ones(5), size=len(x)).T
    ax.stackplot(x, y)
    ax.set_title("Market Share")
    return fig

def story_demographic_analysis():
    fig, ax = plt.subplots()
    y = np.arange(10)
    m = np.random.randint(10, 100, 10)
    f = np.random.randint(10, 100, 10)
    ax.barh(y, m)
    ax.barh(y, -f)
    ax.set_title("Demographics")
    return fig
