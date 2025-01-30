from abc import ABC, abstractmethod
from types import SimpleNamespace

from matplotlib import pyplot as plt
import numpy as np
import torch

from scipy.optimize import minimize_scalar, brentq

def get_default_motif_class(motif):
    
    infinite_motif_classes = {
            '++f': PPF(),
            '+-p': PMP(),
            '+-h': PMH(),
            '-+f': MPF(),
            '-+h': MPH(),
            '--f': MMF()
        }

    return infinite_motif_classes[motif]


class InfinitiveMotif(ABC):

    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.parameters['torch'] = {}
        self.parameters['numpy'] = {}
    
    @abstractmethod
    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        pass

    @abstractmethod
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        pass

    @abstractmethod
    def predict(self, T):
        pass

    def extract_properties_list(self):
        pass

    @abstractmethod
    def num_network_properties(self):
        pass

    def evaluate_from_network(self, T, properties, x0, y0, y0dt, y0dt2):
        self.set_by_network(properties, x0, y0, y0dt, y0dt2)
        return self.predict(T)

    def extract_properties_from_network(self, properties, x0, y0, y0dt, y0dt2):
        self.set_by_network(properties, x0, y0, y0dt, y0dt2)
        return self.cat(self.extract_properties_list(),dim=1)
    
    def evaluate_from_properties(self, T, properties, x0, y0, y0dt, y0dt2):
        self.set_by_properties(properties, x0, y0, y0dt, y0dt2)
        return self.predict(T)

        
    def stack(self, list_to_stack, dim=0):
        if isinstance(list_to_stack[0],torch.Tensor):
            return torch.stack(list_to_stack,dim)
        elif isinstance(list_to_stack[0],np.ndarray):
            return np.stack(list_to_stack,dim)
        else:
            raise Exception("Type not supported")
        
    def exp(self,x):
            if isinstance(x,torch.Tensor):
                return torch.exp(x)
            elif isinstance(x,np.ndarray) or np.isscalar(x):
                return np.exp(x)
            else:
                print(type(x))
                raise Exception("Type not supported")
    def log(self,x):
        if isinstance(x,torch.Tensor):
            return torch.log(x)
        elif isinstance(x,np.ndarray):
            return np.log(x)
        elif np.isscalar(x):
            return np.log(x)
        else:
            raise Exception("Type not supported")
    def cat(self,list_to_cat,dim=0):
        if isinstance(list_to_cat[0],torch.Tensor):
            return torch.cat(list_to_cat,dim)
        elif isinstance(list_to_cat[0],np.ndarray):
            return np.concatenate(list_to_cat,dim)
        else:
            raise Exception("Type not supported")
        
    def pow(self,x,y):
        if isinstance(x,torch.Tensor):
            return torch.pow(x,y)
        elif isinstance(x,np.ndarray):
            return np.float_power(x,y)
        elif np.isscalar(x):
            return np.float_power(x,y)
        else:
            raise Exception("Type not supported")
    def sigmoid(self,x):
        if isinstance(x,torch.Tensor):
            return torch.sigmoid(x)
        elif isinstance(x,np.ndarray):
            return 1/(1+np.exp(-x))
        else:
            raise Exception("Type not supported")
    def min(self,x,y):
        if isinstance(x,torch.Tensor):
            if isinstance(y,float):
                y = torch.ones_like(x) * y
            return torch.minimum(x,y)
        elif isinstance(x,np.ndarray):
            if isinstance(y,float):
                y = np.ones_like(x) * y
            return np.minimum(x,y)
        elif np.isscalar(x):
            return np.minimum(x,y)
        else:
            raise Exception("Type not supported")
    def max(self,x,y):
        if isinstance(x,torch.Tensor):
            return torch.maximum(x,y)
        elif isinstance(x,np.ndarray):
            return np.maximum(x,y)
        elif np.isscalar(x):
            return np.maximum(x,y)
        else:
            raise Exception("Type not supported")
        
    def get_property_names(self):
        return [f'property {i}' for i in range(self.num_network_properties())]
        
    
class PPF(InfinitiveMotif):
    # Ae^(Bx) + Cx^2 + Dx + E
    # y0 = A + E
    # y1 = AB + D
    # y2 = A*B^2 + 2C
    def __init__(self):
        super().__init__()

    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        # self.A = properties[:,[0]]
        self.B = properties[:,[0]]
        self.A = self.min(y0dt / self.B,self.min(y0dt2/(self.B**2),10.0))/10
        self.C = (self.Y2 - self.A*self.B**2)/2
        self.D = self.Y1 - self.A*self.B
        self.E = y0 - self.A
       
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        # self.A = properties[0]
        # self.A = 1.0
        self.B = self.log(2)/properties[0]
        self.A = self.min(y0dt / self.B,self.min(y0dt2/(self.B**2),10.0))/10
        self.C = (self.Y2 - self.A*self.B**2)/2
        self.D = self.Y1 - self.A*self.B
        self.E = y0 - self.A

    def predict(self, T):
        return self.A*self.exp(self.B*(T-self.X0)) + self.C*(T-self.X0)**2 + self.D*(T-self.X0) + self.E

    def extract_properties_list(self):
        return [self.log(2)/self.B]
    
    def num_network_properties(self):
        return 1
    
    def get_property_names(self):
        return ['doubling time']
    
    def second_derivative_vanishes(self):
        return False
    
class PMP(InfinitiveMotif):
    # described by A*log(Bx^2 + Cx + 1)+D
    # y0 = D
    # y1 = A*C
    # y2 = A(2B - C^2) = 0
    def __init__(self):
        super().__init__()

    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.A = properties[:,[0]]
        self.C = self.Y1 / self.A
        self.B = self.C**2 / 2
        self.D = self.Y0
       
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.A = properties[0]/self.log(2.0)
        self.C = self.Y1 / self.A
        self.B = self.C**2 / 2
        self.D = self.Y0

    def predict(self, T):
        return self.A*self.log(self.B*(T-self.X0)**2 + self.C*(T-self.X0) + 1) + self.D

    def extract_properties_list(self):
        return [self.A*self.log(2.0)]
    
    def num_network_properties(self):
        return 1
    
    def get_property_names(self):
        return ['incrementing factor']

    def second_derivative_vanishes(self):
        return True
    
# class PMH(InfinitiveMotif):
#     # described by A/(1+e^(-Bt)) + C
#     # y0 = A/2 + C
#     # y1 = AB/4
#     # y2 = 0
#     def __init__(self):
#         super().__init__()

#     def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
#         self.X0 = x0
#         self.Y0 = y0
#         self.Y1 = y0dt
#         self.Y2 = y0dt2
#         self.A = properties[:,[0]]
#         self.B = (4*y0dt) / self.A
#         self.C = y0 - self.A/2
       
#     def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
#         self.X0 = x0
#         self.Y0 = y0
#         self.Y1 = y0dt
#         self.Y2 = y0dt2
#         ApC = properties[0]
#         self.A = 2*(ApC - y0)
#         self.B = (4*y0dt) / self.A
#         self.C = y0 - self.A/2

#     def predict(self, T):
#         return self.A/(1+self.exp(-self.B*(T-self.X0))) + self.C

#     def extract_properties_list(self):
#         return [self.A+self.C]
    
#     def num_network_properties(self):
#         return 1
    

class PMH(InfinitiveMotif):
    # Described by h = f \circ g where f is given by f(t) = -A/(1+e^(B*t)) + C
    # and g is a cubic spline with three knots that is increasing and concave
    # g(0) = 0
    # f(0) = -A/2 + C
    # f'(0) = AB/4
    # f''(0) = 0
    # lim h = C
    # C = L
    # A = 2*(L-y0)
    # g'(0) = 2*y1/((L-y0)*B)
    # g''(0) = 0
    # g(H) = ln(3)/B
    # H > (ln(3)*(L-y0))/(2*y1)
    # g''(H) = 0
    
    def __init__(self):
        super().__init__()

    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.C = y0 + properties[:,[0]] # The horizontal asymptote needs to be below the y0 value
        self.A = 2*(self.C-y0) # To ensure y0
        self.B = 1
        self.H = self.log(3)*(self.C-y0)/(2*y0dt) + properties[:,[1]]



    def predict_cubic_spline(self, T):
        # We assume that it start from t=0
        # Cubic spline has three knots and is described by t1 and h that describe the position and the value of the second derivative
        # t1 can be any value as long as it is smaller then H. We choose t1 = H/2
        value = self.log(3)/self.B
        g1 = 2*self.Y1/((self.C-self.Y0)*self.B)
        H_threshold_param = 1.5 # it has to be less than 2
        H_threshold = H_threshold_param * value/g1 

        if isinstance(T,torch.Tensor):
            mask_H = torch.where(self.H <= H_threshold,1,0)
        else:
            mask_H = np.where(self.H <= H_threshold,1,0)

        t2 = self.H * mask_H + (1-mask_H) * H_threshold

        h1 = 4*(-self.H*g1+value)/(self.H**2)
        # h2 = 4*(g1**2)*(value - self.H*g1)/(2*self.H*value*g1-value**2)
        h2 = 4*(-self.H*g1+value)/(2*self.H*t2-(t2**2))

        h = h1 * mask_H + h2 * (1-mask_H)

        coeffs1 = [h/(3*t2),0,g1,0]
        coeffs2 = [-h/(3*t2),h,g1-((t2*h)/2),((t2**2)*h)/12]
        first_derivative_end = (t2 * h) / 2 + g1
        value_at_end = t2*(h*t2+4*g1)/4

        if isinstance(T,torch.Tensor):
            g = torch.zeros_like(T)

            mask1 = torch.where(T < t2/2,1,0)
            mask2 = torch.where((T >= t2/2) & (T < t2),1,0)
            mask3 = torch.where(T >= t2,1,0)

            g += (coeffs1[0]*T**3 + coeffs1[1]*T**2 + coeffs1[2]*T + coeffs1[3]) * mask1
            g += (coeffs2[0]*T**3 + coeffs2[1]*T**2 + coeffs2[2]*T + coeffs2[3]) * mask2
            g += ((T - t2) * first_derivative_end + value_at_end) * mask3
        
        else:
            if len(T.shape) == 1:
                T = T.reshape(1,len(T))
         
            mask1 = np.where(T < t2/2,1,0)
            mask2 = np.where((T >= t2/2) & (T < t2),1,0)
            mask3 = np.where(T >= t2,1,0)
            
            g = np.zeros_like(T,dtype=float)
        
            g += (coeffs1[0]*T**3 + coeffs1[1]*T**2 + coeffs1[2]*T + coeffs1[3]) * mask1
            g += (coeffs2[0]*T**3 + coeffs2[1]*T**2 + coeffs2[2]*T + coeffs2[3]) * mask2
            g += ((T - t2) * first_derivative_end + value_at_end) * mask3

            if g.shape[0] == 1:
                g = g.flatten()

            # plt.plot(T.flatten(),g.flatten())
            # plt.scatter([t2.item()/2,t2.item(),self.H.item()],[0.5,value_at_end.item(),value], color='blue')

        return g


       
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.C = properties[0]
        self.H = properties[1] - x0
        self.A = 2*(self.C-y0)
        self.B = 1
       
    def predict(self, T):
        return -self.A/(1+self.exp(self.B*self.predict_cubic_spline(T-self.X0))) + self.C

    def extract_properties_list(self):
        return [self.C, self.H+self.X0]
    
    def num_network_properties(self):
        return 2
    
    def get_property_names(self):
        return ['horizontal asymptote', 'inverse half-life']
    
    def second_derivative_vanishes(self):
        return True
    
class MPF(InfinitiveMotif):
    # -A*log(Bx^2 + Cx + 1)+D
    # y0 = D
    # y1 = -A*C
    # y2 = -A(2B - C^2) = 0
    def __init__(self):
        super().__init__()

    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.A = properties[:,[0]]
        self.C = -y0dt / self.A
        self.B = self.C**2 / 2
        self.D = y0
       
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.A = properties[0]/self.log(2.0)
        self.C = -y0dt / self.A
        self.B = self.C**2 / 2
        self.D = y0
       
    def predict(self, T):
        return -self.A*self.log(self.B*(T-self.X0)**2 + self.C*(T-self.X0) + 1) + self.D

    def extract_properties_list(self):
        return [self.log(2)*self.A]
    
    def num_network_properties(self):
        return 1
    
    def get_property_names(self):
        return ['decrementing factor']
    
    def second_derivative_vanishes(self):
        return True


class MPH(InfinitiveMotif):
    # Described by h = f \circ g where f is given by f(t) = A/(1+e^(B*t)) + C
    # and g is a cubic spline with three knots that is increasing and concave
    # g(0) = 0
    # f(0) = A/2 + C
    # f'(0) = -AB/4
    # f''(0) = 0
    # lim h = C
    # C = L
    # A = 2*(y0 - L)
    # g'(0) = -2*y1/((y0-L)*B)
    # g''(0) = 0  or more generally g''(0) = y2 / (-AB/4)
    # g(H) = ln(3)/B
    # H > (ln(3)*(y0-L))/(-2*y1)
    # g''(H) = 0
    
    def __init__(self):
        super().__init__()

    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.C = y0 - properties[:,[0]] # The horizontal asymptote needs to be below the y0 value
        self.A = 2*(y0 - self.C) # To ensure y0
        self.B = 1
        self.H = self.log(3)*(y0-self.C)/(-2*y0dt) + properties[:,[1]]



    def predict_cubic_spline(self, T):
        # We assume that it start from t=0
        # Cubic spline has three knots and is described by t1 and h that describe the position and the value of the second derivative
        # t1 can be any value as long as it is smaller then H. We choose t1 = H/2
        value = self.log(3)/self.B
        # print(self.Y0)
        # print(self.C)
        g1 = -2*self.Y1/((self.Y0-self.C)*self.B)
        H_threshold_param = 1.5 # it has to be less than 2
        H_threshold = H_threshold_param * value/g1 

        if isinstance(T,torch.Tensor):
            mask_H = torch.where(self.H <= H_threshold,1,0)
        else:
            mask_H = np.where(self.H <= H_threshold,1,0)

        t2 = self.H * mask_H + (1-mask_H) * H_threshold

        h1 = 4*(-self.H*g1+value)/(self.H**2)
        # h2 = 4*(g1**2)*(value - self.H*g1)/(2*self.H*value*g1-value**2)
        h2 = 4*(-self.H*g1+value)/(2*self.H*t2-(t2**2))

        h = h1 * mask_H + h2 * (1-mask_H)

        coeffs1 = [h/(3*t2),0,g1,0]
        coeffs2 = [-h/(3*t2),h,g1-((t2*h)/2),((t2**2)*h)/12]
        first_derivative_end = (t2 * h) / 2 + g1
        value_at_end = t2*(h*t2+4*g1)/4

        if isinstance(T,torch.Tensor):
            g = torch.zeros_like(T)

            mask1 = torch.where(T < t2/2,1,0)
            mask2 = torch.where((T >= t2/2) & (T < t2),1,0)
            mask3 = torch.where(T >= t2,1,0)

            g += (coeffs1[0]*T**3 + coeffs1[1]*T**2 + coeffs1[2]*T + coeffs1[3]) * mask1
            g += (coeffs2[0]*T**3 + coeffs2[1]*T**2 + coeffs2[2]*T + coeffs2[3]) * mask2
            g += ((T - t2) * first_derivative_end + value_at_end) * mask3
        
        else:
            if len(T.shape) == 1:
                T = T.reshape(1,len(T))
         
            mask1 = np.where(T < t2/2,1,0)
            mask2 = np.where((T >= t2/2) & (T < t2),1,0)
            mask3 = np.where(T >= t2,1,0)
            
            g = np.zeros_like(T,dtype=float)
        
            g += (coeffs1[0]*T**3 + coeffs1[1]*T**2 + coeffs1[2]*T + coeffs1[3]) * mask1
            g += (coeffs2[0]*T**3 + coeffs2[1]*T**2 + coeffs2[2]*T + coeffs2[3]) * mask2
            g += ((T - t2) * first_derivative_end + value_at_end) * mask3

            if g.shape[0] == 1:
                g = g.flatten()

            # plt.plot(T.flatten(),g.flatten())
            # plt.scatter([t2.item()/2,t2.item(),self.H.item()],[0.5,value_at_end.item(),value], color='blue')

        return g


       
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        self.C = properties[0]
        self.H = properties[1] - x0
        self.A = 2*(y0 - self.C)
        self.B = 1
       
    def predict(self, T):
        return self.A/(1+self.exp(self.B*self.predict_cubic_spline(T-self.X0))) + self.C

    def extract_properties_list(self):
        return [self.C, self.H+self.X0]
    
    def num_network_properties(self):
        return 2
    
    def get_property_names(self):
        return ['horizontal asymptote', 'half-life']
    
    def second_derivative_vanishes(self):
        return True
    
    # -A/(1+e^(-B(t+C)))+D
    
# class MPH(InfinitiveMotif):
#     # described by L + (y0-L)*((1+C*x)/((1+A*x)**B))
    

#     def __init__(self):
#         super().__init__()
#         self.epsilon = 1e-6 # To avoid division by zero

#     def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
#         self.X0 = x0
#         self.Y0 = y0
#         self.Y1 = y0dt
#         self.Y2 = y0dt2
#         self.L = y0 - properties[:,[0]] # L specifies the asymptote and it needs to below the y0 value
#         # self.B = 1 + properties[:,[1]]*2 # B specifies the steepness of the curve (must be bigger than 1)
#         self.B = 1.1 + 3*(self.sigmoid(properties[:,[1]]) - 0.5) # B specifies the steepness of the curve (must be bigger than 1)
#         self.A = (-2*y0dt)/(-self.B*self.L + self.B*y0 + self.L - y0 + self.epsilon)
#         self.C = (-y0dt * (self.B+1))/(-self.B*self.L + self.B*y0 + self.L - y0 + self.epsilon)

       
#     def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
#         self.X0 = x0
#         self.Y0 = y0
#         self.Y1 = y0dt
#         self.Y2 = y0dt2
#         self.L = properties[0]
#         self.HL = properties[1]
#         self.B = self.get_B_from_half_life()
#         self.A = (-2*y0dt)/(-self.B*self.L + self.B*y0 + self.L - y0 + self.epsilon)
#         # A = (-2*y0dt)/((B-1)*(y0-L)) - should be positive
#         self.C = (-y0dt * (self.B+1))/(-self.B*self.L + self.B*y0 + self.L - y0 + self.epsilon)
       
#     def predict(self, T):
     
#         pred = self.L + (self.Y0-self.L)*((1+self.C*(T-self.X0))/(self.pow((1+self.A*(T-self.X0)),self.B)))
#         # If the prediction is nan or inf then print the values
#         if isinstance(pred,torch.Tensor):
#             if torch.isnan(pred).any() or torch.isinf(pred).any():
#                 org_mask = torch.isnan(pred) | torch.isinf(pred)
#                 # Reduce mask to one dimension
#                 mask = torch.any(org_mask,dim=1)
#                 print(f"L: {self.L[mask]}")
#                 print(f"Y0: {self.Y0[mask]}")
#                 print(f"Y1: {self.Y1[mask]}")
#                 print(f"Y2: {self.Y2[mask]}")
#                 print(f"A: {self.A[mask]}")
#                 print(f"B: {self.B[mask]}")
#                 print(f"C: {self.C[mask]}")
#                 print(f"Denominator: {self.pow((1+self.A*(T-self.X0)),self.B)}")
#                 print(f" T-XO: {(T-self.X0)}")
#                 raise Exception("Prediction is nan or inf")
#         return self.L + (self.Y0-self.L)*((1+self.C*(T-self.X0))/(self.pow((1+self.A*(T-self.X0)),self.B)))

#     def extract_properties_list(self):
#         self.HL = self.get_half_life_from_B()
#         return [self.L, self.HL]
    
#     def num_network_properties(self):
#         return 2
    
#     def get_property_names(self):
#         return ['horizontal asymptote', 'half-life']
    
#     def get_half_life_from_B(self):

#         if isinstance(self.X0,torch.Tensor):
#             HL = torch.zeros(self.X0.shape)
#             for i in range(self.X0.shape[0]):
#                 a = self.A[i].item()
#                 b = self.B[i].item()
#                 c = self.C[i].item()
#                 l = self.L[i].item()
#                 y0 = self.Y0[i].item()
#                 x0 = self.X0[i].item()
#                 y1 = self.Y1[i].item()

#                 def f(t):
#                     return l + (y0-l)*((1+c*t)/(self.pow((1+a*t),b))) - (y0+l)/2
                
                
#                 min_hl = max(0,(y0+l)/(-2*y1))
#                 # hl = minimize_scalar(f,bounds=(min_hl+self.epsilon,1000)).x
#                 if f(min_hl+self.epsilon) <= 0:
#                     print(f"b: {b}, a: {a}, c: {c}, y0-l: {y0-l}, l: {l}, y0: {y0}, x0: {x0}, y1: {y1}")
#                     print(f"min_hl: {min_hl}, f(min_hl+eps): {f(min_hl+self.epsilon)}")
#                 if f(1e6) >= 0:
#                     print(f"b: {b}, a: {a}, c: {c}, y0-l: {y0-l}, l: {l}, y0: {y0}, x0: {x0}, y1: {y1}")
#                     print(f"min_hl: {min_hl}, f(1e6): {f(1e6)}")
#                 hl = brentq(f,min_hl+self.epsilon,1e6)
#                 # print(f"b: {b}, a: {a}, c: {c}, y0-l: {y0-l}, l: {l}, y0: {y0}, x0: {x0}, y1: {y1}")
#                 HL[i,0] = hl + x0
#             return HL
#         else:
#             a = self.A
#             b = self.B
#             c = self.C
#             l = self.L
#             y0 = self.Y0
#             x0 = self.X0
#             y1 = self.Y1
#             def f(t):
#                 return l + (y0-l)*((1+c*(t))/(self.pow((1+a*(t)),b))) - (y0+l)/2
                
#             min_hl = max((y0+l)/(-2*y1),0)
#             hl = brentq(f,min_hl+self.epsilon,1e6)
#             return hl + x0
    
#     def get_B_from_half_life(self):
#         l = self.L
#         y0 = self.Y0
#         x0 = self.X0
#         y1 = self.Y1
#         hl = self.HL
#         hl = hl - x0

#         def f(b):
#             a = (-2*y1)/(-b*l + b*y0 + l - y0 + self.epsilon)
#             c = (-y1 * (b+1))/(-b*l + b*y0 + l - y0 + self.epsilon)
#             return l + (y0-l)*((1+c*(hl))/(self.pow((1+a*(hl)),b))) - (y0+l)/2
        
#         # b = minimize_scalar(f,bounds=(1+self.epsilon,1000)).x
#         b = brentq(f,1+self.epsilon,1000)
#         return b

    

# class MPH(InfinitiveMotif):
#     # described by -A/(1+e^(-Bt)) + C
#     # y0 = -A/2 + C
#     # y1 = -AB/4
#     # y2 = 0
#     def __init__(self):
#         super().__init__()

#     def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
#         self.X0 = x0
#         self.Y0 = y0
#         self.Y1 = y0dt
#         self.Y2 = y0dt2
#         self.A = y0 - properties[:,[0]]
#         self.B = (-4*y0dt) / self.A
#         self.C = y0 + self.A/2
       
#     def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
#         self.X0 = x0
#         self.Y0 = y0
#         self.Y1 = y0dt
#         self.Y2 = y0dt2
#         CmA = properties[0]
#         self.A = 2*(y0 - CmA)
#         self.B = (-4*y0dt) / self.A
#         self.C = y0 + self.A/2
       
#     def predict(self, T):
#         return -self.A/(1+self.exp(-self.B*(T-self.X0))) + self.C

#     def extract_properties_list(self):
#         return [self.C - self.A]
    
#     def num_network_properties(self):
#         return 1
    
#     # -A/(1+e^(-B(t+C)))+D
    
class MMF(InfinitiveMotif):
    # -Ae^(Bx) - Cx^2 - Dx + E
    # y0 = -A + E
    # y1 = -AB - D
    # y2 = -A*B^2 - 2C
    def __init__(self):
        super().__init__()

    def set_by_network(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        # self.A = properties[:,[0]]
        self.B = properties[:,[0]]
        self.A = self.min(-y0dt/self.B,self.min(-y0dt2/self.B**2,2.0))/2
        self.C = -(y0dt2 + self.A*self.B**2)/2
        self.D = -y0dt - self.A*self.B
        self.E = y0 + self.A
       
    def set_by_properties(self, properties, x0, y0, y0dt, y0dt2):
        self.X0 = x0
        self.Y0 = y0
        self.Y1 = y0dt
        self.Y2 = y0dt2
        # self.A = properties[0]
        self.B = self.log(2)/properties[0]
        self.A = self.min(-y0dt/self.B,self.min(-y0dt2/self.B**2,2.0))/2
        self.C = -(y0dt2 + self.A*self.B**2)/2
        self.D = -y0dt - self.A*self.B
        self.E = y0 + self.A
       
    def predict(self, T):
        return -self.A*self.exp(self.B*(T-self.X0)) - self.C*(T-self.X0)**2 - self.D*(T-self.X0) + self.E

    def extract_properties_list(self):
        return [self.log(2)/self.B]
    
    def num_network_properties(self):
        return 1
    
    def get_property_names(self):
        return ['negative doubling time']
    
    def second_derivative_vanishes(self):
        return False


    

    





    