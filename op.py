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
        result = minimize(objective_function, x0, method=method)
        return result
    except Exception as e:
        print("Optimization failed:", str(e))

# Usage example
x0 = np.array([0, 0])

result = optimize_objective_function(x0)
if result is not None:
    print("Optimum values:", result.x)
    print("Minimum value:", result.fun)
