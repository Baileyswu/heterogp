#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("../src/")
import numpy as np
import pylab as plt
plt.style.use('ggplot')
import astropy.units as au
import os

import gpflow as gp
from heterogp.latent import Latent
from gpflow import settings
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
import tensorflow as tf

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import mpl_toolkits.mplot3d.axes3d as p3


# # Some helper functions

# In[ ]:


from gpflow.actions import Loop, Action
from gpflow.training import AdamOptimizer

class PrintAction(Action):
    def __init__(self, model, text):
        self.model = model
        self.text = text
        
    def run(self, ctx):
        if ctx.iteration % 200 == 0:
            likelihood = ctx.session.run(self.model.likelihood_tensor)
            logging.warning('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood))
    #         logging.warning(self.model)
        
def run_with_adam(model, lr,iterations, callback=None):
    
    adam = AdamOptimizer(lr).make_optimize_action(model)
    
    actions = [adam]#natgrad,
    actions = actions if callback is None else actions + [callback]

    Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())


# In[ ]:


import pandas as pd
def draw(X, ys):
    plt.figure(figsize=(6, 6))
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(
        X[:,0].reshape(-1,1),
        X[:,1].reshape(-1,1),
        cmap="coolwarm", 
        c=ys.reshape(-1,1))
    plt.show()

    table = pd.DataFrame(ys)
    table.describe()
    table[0].hist()
    
def acc_rate(ystar):
    ypred = 1.*(ystar > 0.5)
    acc = 1.-np.count_nonzero(ypred - Ytest) / Ytest.shape[0]
    print('acc_rate', acc)


# # Define some data with input-dependent noise

# In[ ]:


def func(X):
    return 1.*(X[:,0]**2 + (X[:,1]-1)**2 <= 0.5)

def func_with_noise(X):
    return 1.*(X[:,0]**2 + (X[:,1]-1)**2 + 0.1*np.random.normal(size=X.shape[0]) <= 0.5)

N = 900
Nn = 30

k,b=np.mgrid[-1:1:30j,0:2:30j] # Nn
f_kb=1.*(k**2 + (b-1)**2 + 0.1*np.random.normal(size=k.shape) <= 0.5)

hk = np.hstack(k).reshape(-1, 1)
hb = np.hstack(b).reshape(-1, 1)
Xtrain = np.hstack((hk, hb))
Ytrain = np.hstack(f_kb).reshape(-1, 1)

draw(Xtrain, Ytrain)

N, D = Xtrain.shape
Xtest = np.random.random_sample((int(N*0.4), 2))*2+[-1, 0]
Ytest = func(Xtest).reshape(-1,1)
draw(Xtest, Ytest)


# # Define the HGP model and train
# 
# We will:
#   - Define the latent GP that models the noise
#   - Define heteroscedastic likelihood which uses the above latent
#   - Define the HGP which has another independent latent modelling the
#   underlying function
#   - Finally, train with Adam and plot the results

# In[ ]:


from heterogp.likelihoods import HeteroscedasticGaussian
from heterogp.hgp import HGP

settings.numerics.jitter_level=1e-6
iterations = 5000
num_inducing = min(100, N/10) # TODO

from scipy.cluster.vq import kmeans
Z = kmeans(Xtrain, num_inducing)[0] 
# Z = np.linspace(-2,2,100)[:,None]

with tf.Session(graph=tf.Graph()) as sess:
    with gp.defer_build():       
        
        # Define the (log) noise latent
        mean = gp.mean_functions.Constant(np.log(0.5))
        kern = gp.kernels.RBF(D) # TODO
        log_noise_latent = Latent(Z, mean, kern, num_latent=1, whiten=False, name=None)
        # Define the likelihood
        likelihood = HeteroscedasticGaussian(log_noise_latent)
        log_noise_latent
        # Define the underlying GP mean and kernel
        mean = gp.mean_functions.Zero()
        kernel = gp.kernels.RBF(D) # TODO
        # Create the HGP (note the slightly different order from SVGP)
        model = HGP(Xtrain, Ytrain, Z, kernel, likelihood, 
                     mean_function=mean, 
                     minibatch_size=500,
                     num_latent = 1, 
                     num_samples=1,
                     num_data=None,
                     whiten=False)
        model.compile()
    from timeit import default_timer
    t0 = default_timer()
    run_with_adam(model,1e-3,iterations,PrintAction(model,"Adam"))
    print(default_timer() - t0)
    # Predictions uses stochastic sampling and produces 
    # [num_samples,N,D] shape output
    ystar,varstar = model.predict_y(Xtest, 100)
    # For plotting the noise
    hetero_noise = model.likelihood.compute_hetero_noise(Xtest, 100)


# In[ ]:


acc_rate(ystar.mean(0))
draw(Xtest, ystar.mean(0))


# In[ ]:


m = gp.models.SVGP(
    Xtrain, Ytrain, kern=gp.kernels.RBF(2),
    likelihood=gp.likelihoods.Bernoulli(), Z=Z)
# Initially fix the hyperparameters.
m.feature.set_trainable(False)
gp.train.ScipyOptimizer().minimize(m, maxiter=iterations)

# Unfix the hyperparameters.
m.feature.set_trainable(True)
gp.train.ScipyOptimizer(options=dict(maxiter=iterations)).minimize(m)
ystar,varstar = m.predict_y(Xtest, 100)


# In[ ]:


acc_rate(ystar)
draw(Xtest, ystar[:,0])


# In[ ]:


m = gp.models.SGPR(
    Xtrain, Ytrain, kern=gp.kernels.RBF(2), Z=Z)
# Initially fix the hyperparameters.
m.feature.set_trainable(False)
gp.train.ScipyOptimizer().minimize(m, maxiter=iterations)

# Unfix the hyperparameters.
m.feature.set_trainable(True)
gp.train.ScipyOptimizer(options=dict(maxiter=iterations)).minimize(m)
ystar,varstar = m.predict_y(Xtest, 100)


# In[ ]:


acc_rate(ystar)
draw(Xtest, ystar[:,0])

