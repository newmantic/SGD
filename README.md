# SGD


Stochastic Gradient Descent is an iterative optimization algorithm used to minimize an objective function, commonly in the context of machine learning and statistics. It is particularly useful for optimizing loss functions in training models.


Objective Function: Given a dataset of N samples, the objective function to minimize can be expressed as:
J(θ) = (1/N) ∑ (i=1 to N) L(θ; x_i, y_i)
where:
J(θ) is the cost function.
L(θ; x_i, y_i) is the loss function for the i-th sample.
θ represents the parameters of the model.
(x_i, y_i) are the features and target value of the i-th sample.


Initialize the model parameters θ (weights) to small random values or zeros.

Iterate:
For each epoch (iteration over the entire dataset):
Shuffle the training dataset to ensure randomness.

For each training example (x_i, y_i) in the dataset:

Compute the prediction:
y_hat_i = h(θ; x_i)

Compute the loss (error):
error_i = y_hat_i - y_i

Update the parameters θ using the gradient of the loss function:
θ = θ - η * ∇L(θ; x_i, y_i)
where:
η is the learning rate (a small positive constant).
∇L(θ; x_i, y_i) is the gradient of the loss function with respect to the parameters.

Convergence Check:
The algorithm continues to iterate until the change in the cost function J(θ) is smaller than a predefined tolerance level, or until a maximum number of epochs is reached.

Learning Rate:
The learning rate η controls the size of the steps taken towards the minimum. If η is too large, the algorithm may overshoot the minimum; if η is too small, the convergence may be slow.


Pros
Efficiency: SGD is computationally efficient as it updates parameters using a single example, which is faster than batch gradient descent that uses the entire dataset.
Online Learning: It can handle streaming data and is well-suited for large datasets.

Cons
Noisy Updates: The updates are noisy due to the use of a single training example, which may lead to oscillations.
Convergence Issues: It may converge to a local minimum rather than the global minimum, especially if the loss function is not convex.
