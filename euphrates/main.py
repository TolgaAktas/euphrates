import os
import numpy as np

training_epochs = 1000
momentum_betas = (0.9,0.999)
init_learning_rate = 3e-5

num_inference_steps = 25
num_diffusion_steps = 1000
beta_noise = np.linspace(0.0001,0.02,num_diffusion_steps)

eta_degradation = 1e-4
lambda_regularizer = 0.5

