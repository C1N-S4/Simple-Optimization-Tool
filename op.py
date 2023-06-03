import numpy as np
from scipy.optimize import minimize

def objective_function(x):
    return (x[0] - 4)**2 + (x[1] - 3)**2

def optimize_objective_function(x0):
    if not isinstance(x0, np.ndarray):
        raise TypeError("Initial guess x0 must be a NumPy array.")
    if x0.shape != (2,):
        raise ValueError("Initial guess x0 must have shape (2,).")
    
    valid_methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC']
    method = 'BFGS'  # Default method
    
    if method not in valid_methods:
        raise ValueError("Invalid optimization method.")
    
    try:
        result = minimize(objective_function, x0, method=method, callback=print_metrics)
        return result
    except Exception as e:
        print("Optimization failed:", str(e))

# Yeni fonksiyon
def print_metrics(xk):
    global iteration_count, minimum_values
    minimum_value = objective_function(xk)
    minimum_values.append(minimum_value)
    iteration_count += 1
    print("Iteration:", iteration_count)
    print("Current minimum value:", minimum_value)
    print("Current x:", xk)
    print()

# Usage example
x0 = np.array([0, 0])
iteration_count = 0
minimum_values = []

result = optimize_objective_function(x0)
if result is not None:
    print("Optimum values:", result.x)
    print("Minimum value:", result.fun)
    print("Iteration count:", iteration_count)
    print("Minimum values per iteration:", minimum_values)
