### Import useful packages
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns; sns.set()
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

###
### This section of the code generates the training data for fitting the Gaussian Process Regression model. 
###
x_end = 2                    ### End point of the training set interval 
n_train = 200                ### Number of extracted training set data points 
x_n = np.linspace(start = 0, stop = x_end, num = n_train)       ### Generating the actual data points 

### Define a simple function f(x)
def f(x_n):
    f = np.sin((2 * np.pi) * x_n)        ### f(x) = sin(2πx)
    return (f)

### Store results in variable f_x
f_x = f(x_n)

### Adding noise to the functional evaluations 
sigma_noise = 0.1           ### Sigma noise is the standard deviation of the normal distribution from which the random numbers are sampled
epsilon_n = np.random.normal(loc = 0, scale = sigma_noise, size = n_train)   ### Adding the noise to each training datapoint 
y_n = f_x + epsilon_n       ### y values of the training set is now the function + error in the training dataset

### Visualise the training data with a plot
fig, ax = plt.subplots(figsize=(12, 5))
sns.scatterplot(x = x_n, y = y_n, label = "training data", ax = ax);            ### Scatterplot of the training set datapoints with added noise
sns.lineplot(x = x_n, y = f_x, color = "red", label = "f(x)", ax = ax);         ### Lineplot of the noise-free function
ax.set(title ="Training data for GPR model")
ax.legend(loc = "upper right")
ax.set(xlabel = "x", ylabel = "y")
plt.show()

###
### This section of the code generates the test data 
###
n_test = n_train + 300
x_star = np.linspace(start = 0, stop = (x_end + 1), num = n_test)

### Reshape the training and test datasets to fit the GPR
d = 1;          ### Specifies the dimensionality of the problem, in this case 1D 
X_n = x_n.reshape(n_train, d)
X_star = x_star.reshape(n_test, d)

###
### This section of the code defines the type of kernel used and the Gaussion Process Regression model
###
def gprPrediction(l, sigma_f, sigma_n, X_n, y_n, X_star):
    ### Define the kernel 
    kernel = ConstantKernel(constant_value = sigma_f, constant_value_bounds = (1e-2, 1e2)) * RBF(length_scale = l, length_scale_bounds = (1e-2, 1e2))
    ### Gaussian Process model 
    gpr = GaussianProcessRegressor(kernel = kernel, alpha = sigma_n ** 2, n_restarts_optimizer = 10, )
    ### Fit the Gaussian Process Regression model 
    gpr.fit(X_n, y_n)
    ### Make the prediction on the test set 
    y_pred = gpr.predict(X_star)
    return y_pred, gpr;

### Set hyperparameters for the kernel function 
l_init = 1
sigma_f_init = 3
sigma_n = 1

y_pred, gpr = gprPrediction(l_init, sigma_f_init, sigma_n, X_n, y_n, X_star)

# Generate samples from posterior distribution. 
y_hat_samples = gpr.sample_y(X_star, n_samples=n_test)
# Compute the mean of the sample. 
y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
# Compute the standard deviation of the sample. 
y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()

fig, ax = plt.subplots(figsize=(15, 8))
# Plotting the training data.
sns.scatterplot(x=x_n, y=y_n, label='training data', ax=ax);
# Plot the functional evaluation
sns.lineplot(x=x_star, y=f(x_star), color='red', label='f(x)', ax=ax)
# Plot corridor. 
ax.fill_between(x=x_star, y1=(y_hat - 2*y_hat_sd), y2=(y_hat + 2*y_hat_sd), color='green',alpha=0.3, label='Credible Interval')
# Plot prediction. 
sns.lineplot(x=x_star, y=y_pred, color='green', label='pred')

# Labeling axes
ax.set(title='Gaussian Process Regression')
ax.legend(loc='lower left');
ax.set(xlabel='x', ylabel='')
plt.show()