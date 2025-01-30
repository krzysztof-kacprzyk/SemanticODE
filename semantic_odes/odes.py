from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import odeint
import pandas as pd
from scipy.interpolate import UnivariateSpline
class ODEBase(ABC):

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"ODE({self.name})"

    @abstractmethod
    def solve(self, y0, t, **kwargs):
        pass

class SimpleLogisticGrowthODE(ODEBase):
    
    def __init__(self):
        super().__init__("simple_logistic_growth")

    def solve(self, y0, t, r=1, K=1):
        def logistic_equation(y, t, r, K):
            return r * y * (1 - y / K)

        solution = odeint(logistic_equation, y0, t, args=(r, K))
        return solution.flatten()
    
class ExponentialODE(ODEBase):
    
    def __init__(self):
        super().__init__("exponential")

    def solve(self, y0, t, r=1):
        def exponential_equation(y, t, r):
            return r * y

        solution = odeint(exponential_equation, y0, t, args=(r,))
        return solution.flatten()
    
class HeatODE(ODEBase):

    def __init__(self):
        super().__init__("heat")
        df = pd.read_csv('data/copper.csv')
        df.sort_values('x', inplace=True)
        spl = UnivariateSpline(df['x'], df['y'])
        self.spl = spl

    def solve(self, y0, t):
        def heat_transfer(T, t, T_ambient):
            return -self.spl(T) * 0.5 * (T - T_ambient)
        
        solution = odeint(heat_transfer, y0, t, args=(40.0,))
        return solution.flatten()

class DuffingODE(ODEBase):

    def __init__(self):
        super().__init__("duffing")

    def solve(self, y0, t, delta=0.1, alpha=-1.0, beta=1.0, gamma=0.3, omega=0.5):
        y0 = [y0, 0.0]
        def duffing_eq(y, t, delta, alpha, beta, gamma, omega):
            x, v = y
            dxdt = v
            dvdt = gamma * np.cos(omega * t) - delta * v - alpha * x - beta * x ** 3
            return [dxdt, dvdt]

        solution = odeint(duffing_eq, y0, t, args=(delta, alpha, beta, gamma, omega))[:,0]
        return solution.flatten()
    
