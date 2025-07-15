import matplotlib.pyplot as plt 
import numpy as np 
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

### Define a range of sampled x values
x = np.linspace(0, 100, 200)

### Define a simple function 
y_true = np.sin(2 * x)

### Generate noise with the same size as that of the sampled data
noise = np.random.normal(0, 0.1, x.shape)

### Generate the new function, including noise 
y_noise = y_true + noise 

### Plot the two functions to see how noise influences the datapoints
fig, axs = plt.subplots(1,2, figsize=(12, 5))

axs[0].plot(x, y_true)
axs[0].set_title("Simple generated function")
axs[0].set(xlabel = "x", ylabel = "y = sin(2x)")

axs[1].plot(x, y_noise)
axs[1].set_title("Simple generated function with noise")
axs[1].set(xlabel = "x", ylabel = "y = sin(2x)")

plt.show()

### Reshape x to fit 2D arrays 
X = x.reshape(-1, 1)

### Set up the kernels -> RBF for smoothness and WhiteKernel for noise 
kernel  = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

### Create and fit the GPR model 
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X, y_noise)

### Predict the data
x_pred = np.linspace(0, 100, 2000).reshape(-1, 1)
y_pred, y_std = gpr.predict(x_pred, return_std=True)

### Plot results 
plt.figure(figsize=(10, 5))
plt.scatter(x, y_noise, color="red", label="Noisy Data")
plt.plot(x, y_true, label="True function values", color="blue", linestyle="--")
plt.plot(x_pred, y_pred, label="GPR mean", color="black")
plt.fill_between(
    x_pred.ravel(),
    y_pred - 2*y_std, y_pred + 2*y_std,
    alpha=0.2, color="grey", label="95% confidence interval"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("GPR fit to sin(2x) with noise")
plt.legend()
plt.tight_layout()
plt.show()