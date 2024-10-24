import numpy as np

class BaseFunction:
    def __init__(self, f, x_lower, x_upper):
        self.f = f
        self.x_lower = np.array(x_lower)
        self.x_upper = np.array(x_upper)
    
    def evaluate(self, x):
        x = np.array(x)
        
        # Apply control mechanisms
        x_controlled = np.clip(x, self.x_lower, self.x_upper)
        
        # Evaluate and return f(x)
        return self.f(x_controlled)
    
