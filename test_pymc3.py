from __future__ import print_function
import pymc3 as pm

if __name__=='__main__':
    
    print('Running on PyMC3 v{}'.format(pm.__version__))

    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize random number generator
    np.random.seed(123)

    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2
    
    # Simulate outcome variable
    Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma
    
    basic_model = pm.Model()

    with basic_model:

        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=10, shape=2)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        # Expected value of outcome
        mu = alpha + beta[0]*X1 + beta[1]*X2
        
        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

    with basic_model:
        # draw 500 posterior samples
        trace = pm.sample(500)

    print(trace['alpha'][-5:])
