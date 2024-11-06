import numpy as np
import matplotlib.pyplot as plt
# Function y = (x + 3)^2
def function(x):
 return (x + 3)**2
# Derivative of the function: dy/dx = 2(x + 3)
def derivative(x):
 return 2 * (x + 3)
# Gradient Descent Algorithm
def gradient_descent(starting_x, learning_rate, num_iterations):
 x = starting_x
 x_values = [x] # Store x values for visualization
 for i in range(num_iterations):
  grad = derivative(x)
  x = x - learning_rate * grad # Update x
  x_values.append(x)
  # Print progress at each step
  print(f"Iteration {i+1}: x = {x:.5f}, y = {function(x):.5f}")
 return x, x_values
# Parameters for gradient descent
starting_x = 2 # Starting point
learning_rate = 0.1 # Learning rate
num_iterations = 20 # Number of iterations
# Perform gradient descent
min_x, x_values = gradient_descent(starting_x, learning_rate, num_iterations)
print(f"\nLocal minima occurs at x = {min_x:.5f}, y = {function(min_x):.5f}")
# Plotting the function and the gradient descent steps
x_range = np.linspace(-10, 4, 100)
y_range = function(x_range)
plt.plot(x_range, y_range, label='y = (x + 3)^2')
plt.scatter(x_values, function(np.array(x_values)), color='red', label='Gradient Descent Steps')
plt.title("Gradient Descent to Find Local Minima")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
