import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def grid_posterior(prior, likelihood_func, data, grid):
    likelihood = likelihood_func(grid, data)
    unnormalized_posterior = prior * likelihood
    posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
    return posterior

def medical_test_bayes(prevalence, sensitivity, specificity):
    p_disease = prevalence
    p_no_disease = 1 - prevalence
    p_pos_given_d = sensitivity
    p_pos_given_nd = 1 - specificity
    
    p_pos = (p_pos_given_d * p_disease) + (p_pos_given_nd * p_no_disease)
    return (p_pos_given_d * p_disease) / p_pos

def sequential_update(prior, likelihood_func, observations, grid):
    posteriors = []
    current_prior = prior
    for obs in observations:
        posterior = grid_posterior(current_prior, likelihood_func, obs, grid)
        posteriors.append(posterior)
        current_prior = posterior
    return posteriors

def beta_binomial_update(alpha, beta, successes, failures):
    return alpha + successes, beta + failures

def plot_prior_likelihood_posterior(prior, likelihood, posterior, grid):
    plt.plot(grid, prior)

def plot_sequential_updates(posteriors, grid, labels):
    pass
