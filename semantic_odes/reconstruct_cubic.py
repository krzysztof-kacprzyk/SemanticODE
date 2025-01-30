import time
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from semantic_odes.infinite_motifs import *

def which_cubic(x, knots_x):
    n_cubic = len(knots_x) - 1
    for i in range(n_cubic):
        if x >= knots_x[i] and x <= knots_x[i+1]: # both equalities are needed to account for edges
            return i
    return -1




class Knot:

    def __init__(self, x=None, y=None, transition_point=None):
        self.x = x
        self.y = y
        self.transition_point = transition_point


class WrappedCubicSpline:

    def __init__(self, semantic_representation, epsilon=1e-9, epsilon_ratio=1e-6):
        # print(semantic_representation)
        self.composition = semantic_representation.composition
        self.transition_points = semantic_representation.coordinates_finite_composition # does not include the transition point at infinity
        self.parameters = None
        self.epsilon = epsilon
        self.epsilon_ratio = epsilon_ratio
        self.t_range = semantic_representation.t_range
        self.derivative_start = semantic_representation.derivative_start
        self.derivative_end = semantic_representation.derivative_end
        self.second_derivative_end = semantic_representation.second_derivative_end

        self.factor = np.max(self.transition_points[:,1]) - np.min(self.transition_points[:,1])
        self.problem = False

        if self.composition[-1][-1] != 'c':
            # it is an infinite composition
            self.composition_finite_part = self.composition[:-1]
            self.infinite_motif = self.composition[-1]
            self.infinite_composition = True
        else:
            # it is a finite composition
            self.composition_finite_part = self.composition
            self.infinite_motif = None
            self.infinite_composition = False
    
        # construct knots
        self.knots = []
        self.additional_knot_ids = []

        for i, point in enumerate(self.transition_points):
            x = point[0]
            y = point[1]
            self.knots.append(Knot(x=x,y=y,transition_point=self.type_of_transition_point(i)))
            if i != len(self.transition_points) - 1:
            # We do not add knots after the last transition point
                self.knots.append(Knot())
                self.additional_knot_ids.append(len(self.knots)-1)    

        self.n_cubic = len(self.knots) - 1

        # second derivative sign across the cubics
        self.second_derivative_sign = self.calculate_second_derivative_signs_for_cubics()

        # first derivative signs across the cubics
        self.first_derivative_sign = self.calculate_first_derivative_signs_for_cubics()      

        self.local_extrema_knots = [i for i, k in enumerate(self.knots) if k.transition_point == 'max' or k.transition_point == 'min']

        self.knots_with_1st_derivative_fixed = [0] + self.local_extrema_knots + [len(self.knots)-1]
        self.knots_with_2nd_derivative_fixed = [
            i for i, k in enumerate(self.knots) if (k.transition_point is not None and k.transition_point != 'inflection') and k.transition_point != 'end'
            ] # second derivative is fixed at the inflection points and at the end
        
        self.fake_knots = [i for i, k in enumerate(self.knots) if k.transition_point is None]

        self.fake_knots_y_determined = []
        self.fake_knots_y_undetermined = []

        # A fake knot before a fixed first derivative is determined.
        # Eiter by the previous extremum or by the first derivative at the start
        for knot_index in self.knots_with_1st_derivative_fixed:
            if knot_index != 0:
                if self.knots[knot_index-1].transition_point is None:
                    self.fake_knots_y_determined.append(knot_index-1)
        # for i, knot_index in enumerate(self.local_extrema_knots):
        #     if i % 2 == 0:
        #         if self.knots[knot_index-1].transition_point is None:
        #             self.fake_knots_y_determined.append(knot_index-1)
        if not self.infinite_composition and len(self.composition) > 1:
            # We have a derivative specified at the end
            # That means the fake knot before it is determined
            self.fake_knots_y_determined.append(len(self.knots)-2)

        self.fake_knots_y_undetermined = []
        for i, knot_index in enumerate(self.fake_knots):
            if self.knots[knot_index].transition_point is None:
                if knot_index not in self.fake_knots_y_determined:
                    self.fake_knots_y_undetermined.append(knot_index)

        
    def calculate_first_derivative_signs_for_cubics(self):
        first_derivative_sign = []
        for cubic_id in range(self.n_cubic):
            comp_id = self.get_motif_id_for_cubic_id(cubic_id)
            if self.composition[comp_id][0] == "+":
                first_derivative_sign.append(1)
            else:
                first_derivative_sign.append(-1)
        return first_derivative_sign
    
    def calculate_second_derivative_signs_for_cubics(self):
        second_derivative_sign = []
        for cubic_id in range(self.n_cubic):
            comp_id = self.get_motif_id_for_cubic_id(cubic_id)
            if self.composition[comp_id][1] == "+":
                second_derivative_sign.append(1)
            else:
                second_derivative_sign.append(-1)
        return second_derivative_sign
    

    def type_of_transition_point(self, ind):
        composition = self.composition_finite_part
        if ind == 0:
            return 'start'
        elif ind == len(composition):
            # We treat the last finite transition point as the end
            return 'end'
        else:
            if (composition[ind-1][:2] == "++") and (composition[ind][:2] == "+-"):
                return 'inflection'
            elif (composition[ind-1][:2] == "+-") and (composition[ind][:2] == "++"):
                return 'inflection'
            elif (composition[ind-1][:2] == "+-") and (composition[ind][:2] == "--"):
                return 'max'
            elif (composition[ind-1][:2] == "-+") and (composition[ind][:2] == "++"):
                return 'min'
            elif (composition[ind-1][:2] == "-+") and (composition[ind][:2] == "--"):
                return 'inflection'
            elif (composition[ind-1][:2] == "--") and (composition[ind][:2] == "-+"):
                return 'inflection'
            else:
                raise ValueError('Unknown transition point type')

            

    def set_parameters(self, parameters):

        self.parameters = parameters

        self.problem = False

        # Scale the parameters
        n_additional_knots = len(self.additional_knot_ids)
        self.parameters[n_additional_knots:] = self.parameters[n_additional_knots:] * self.factor



        self.cached_knot_y_2nd = {}
        self.cached_slope_2nd = {}
        self.cached_intersect_2nd = {}
        self.cached_knot_y_1st = {}
        self.cached_knot_y_0th = {}

        self.propagate_cs()
        self.propagate_ds()

    def is_c_needed(self):
        return len(self.local_extrema_knots) == 0

    def propagate_cs(self):
        self.cs = []
       
        c0 = self.derivative_start
        self.cs.append(c0)

        for index in range(self.n_cubic - 1):
            connecting_knot_id = index + 1
            connecting_knot_x = self.get_knot_x(connecting_knot_id)
            value_at_the_left_of_previous_cubic = (self.get_slope_2nd(index)*connecting_knot_x**2)/2 + self.get_intersect_2nd(index)*connecting_knot_x + self.cs[-1]
            new_c = value_at_the_left_of_previous_cubic - (self.get_slope_2nd(index+1)*connecting_knot_x**2)/2 - self.get_intersect_2nd(index+1)*connecting_knot_x
            self.cs.append(new_c)


    def propagate_ds(self):
        self.ds = []
        first_knot_id = 0
        first_knot_x = self.get_knot_x(first_knot_id)
        first_knot_y = self.knots[first_knot_id].y
        cubic_id = 0
        d0 = first_knot_y -(self.get_slope_2nd(cubic_id)*first_knot_x**3)/6 - self.get_intersect_2nd(cubic_id)*first_knot_x**2/2 - self.cs[cubic_id]*first_knot_x
        self.ds.append(d0)
        # Go through all the cubics to the right of cubic_id
        for cubic_id in range(1,self.n_cubic):
            connecting_knot_id = cubic_id
            connecting_knot_x = self.get_knot_x(connecting_knot_id)
            value_at_the_right_of_previous_cubic = (self.get_slope_2nd(cubic_id-1)*connecting_knot_x**3)/6 + self.get_intersect_2nd(cubic_id-1)*connecting_knot_x**2/2 + self.cs[cubic_id-1]*connecting_knot_x + self.ds[-1]
            new_d = value_at_the_right_of_previous_cubic -(self.get_slope_2nd(cubic_id)*connecting_knot_x**3)/6 - self.get_intersect_2nd(cubic_id)*connecting_knot_x**2/2 - self.cs[cubic_id]*connecting_knot_x
            self.ds.append(new_d)

    def get_knot_x(self, knot_id):
        if knot_id in self.additional_knot_ids:
            ratio = self.parameters[self.additional_knot_ids.index(knot_id)]
            return self.knots[knot_id-1].x + ratio*(self.knots[knot_id+1].x - self.knots[knot_id-1].x)
        else:
            return self.knots[knot_id].x
        
    def get_knot_y_2nd(self, knot_id):
        # print(f"Getting knot y 2nd for knot {knot_id}")
        if knot_id in self.cached_knot_y_2nd:
            return self.cached_knot_y_2nd[knot_id]
        elif knot_id in self.knots_with_2nd_derivative_fixed:
            answer = self.parameters[len(self.additional_knot_ids) + self.knots_with_2nd_derivative_fixed.index(knot_id)]
            self.cached_knot_y_2nd[knot_id] = answer
        elif knot_id in self.fake_knots_y_undetermined:
            answer = self.parameters[len(self.additional_knot_ids) + len(self.knots_with_2nd_derivative_fixed) + self.fake_knots_y_undetermined.index(knot_id)]
            self.cached_knot_y_2nd[knot_id] = answer
        elif knot_id in self.fake_knots_y_determined:

            # we need to find the previous prescribed derivative point
            # we need to find the next prescribed derivative point
            # we need to know the values of the derivatives at these points
            # we need to calculate the signed are between the previous prescribed derivative point and the point before the fake knot

            next_derivative_point = knot_id+1
            prev_derivative_point = self.knots_with_1st_derivative_fixed[self.knots_with_1st_derivative_fixed.index(next_derivative_point)-1]
            
            if prev_derivative_point == 0:
                prev_derivative_value = self.derivative_start
            else:
                prev_derivative_value = 0
            
            if next_derivative_point == len(self.knots)-1:
                next_derivative_value = self.derivative_end
            else:
                next_derivative_value = 0
            
            # We need to calculate a signed area from the previous extremum to the point before the fake knot
            area = 0
            for i in range(prev_derivative_point, knot_id-1):
                # the area is described by a trapezium
                # the two bases are the x-coordinates of the two points
                a = self.get_knot_y_2nd(i)
                b = self.get_knot_y_2nd(i+1)
                # the height
                h = self.get_knot_x(i+1) - self.get_knot_x(i)
                area += (a+b)*h/2
            
            # prev_derivative_value + area + trapezium_1 + trapezium_2 = next_derivative_value
            # trapezium_1 = knot_id-1 to knot_id 
            # trapezium_2 = knot_id to next_derivative_point
            h1 = self.get_knot_x(knot_id) - self.get_knot_x(knot_id-1)
            h2 = self.get_knot_x(next_derivative_point) - self.get_knot_x(knot_id)
            a = self.get_knot_y_2nd(knot_id-1)
            c = self.get_knot_y_2nd(next_derivative_point)
            # print(f"Next derivative value: {next_derivative_value}")
            # print(f"Prev derivative value: {prev_derivative_value}")
            b = (2*next_derivative_value - 2*prev_derivative_value - 2*area - a*h1 - c*h2)/(h1+h2)
            
            answer = b
            # However, we need to make sure that the sign of the second derivative is correct
            if self.second_derivative_sign[knot_id] == 1:
                if answer < 0:
                    self.problem = True
                answer = max(answer,0)
            else:
                if answer > 0:
                    self.problem = True
                answer = min(answer,0)
            self.cached_knot_y_2nd[knot_id] = answer
        elif self.knots[knot_id].transition_point == 'end':
            self.cached_knot_y_2nd[knot_id] = self.second_derivative_end
        elif self.knots[knot_id].transition_point == 'inflection':
            #it's an inflection point
            self.cached_knot_y_2nd[knot_id] = 0.0
        return self.cached_knot_y_2nd[knot_id]
        
    def get_knot_y_1st(self, knot_id, side='right'):
        if (knot_id, side) in self.cached_knot_y_1st:
            return self.cached_knot_y_1st[(knot_id, side)]
        if side == 'right':
            offset = 0
        else:
            offset = 1
        answer = (self.get_slope_2nd(knot_id-offset)*self.get_knot_x(knot_id)**2)/2 + self.get_intersect_2nd(knot_id-offset)*self.get_knot_x(knot_id) + self.cs[knot_id-offset]
        self.cached_knot_y_1st[(knot_id, side)] = answer
        return answer
    
    def get_knot_y_0th(self, knot_id, side='right'):
        if (knot_id, side) in self.cached_knot_y_0th:
            return self.cached_knot_y_0th[(knot_id, side)]
        if side == 'right':
            offset = 0
        else:
            offset = 1
        answer = (self.get_slope_2nd(knot_id-offset)*self.get_knot_x(knot_id)**3)/6 + self.get_intersect_2nd(knot_id-offset)*self.get_knot_x(knot_id)**2/2 + self.cs[knot_id-offset]*self.get_knot_x(knot_id) + self.ds[knot_id-offset]
        self.cached_knot_y_0th[(knot_id, side)] = answer
        return answer
    
    def get_slope_2nd(self, cubic_id):
        if cubic_id in self.cached_slope_2nd:
            return self.cached_slope_2nd[cubic_id]
        numerator = self.get_knot_y_2nd(cubic_id+1) - self.get_knot_y_2nd(cubic_id)
        denominator = self.get_knot_x(cubic_id+1) - self.get_knot_x(cubic_id)
        if np.abs(denominator) < 1e-6:
            answer = numerator*1e6
        else:
            answer = numerator/denominator
        self.cached_slope_2nd[cubic_id] = answer
        return answer
       
    def get_intersect_2nd(self, cubic_id):
        if cubic_id in self.cached_intersect_2nd:
            return self.cached_intersect_2nd[cubic_id]
        answer = self.get_knot_y_2nd(cubic_id) - self.get_slope_2nd(cubic_id)*self.get_knot_x(cubic_id)
        self.cached_intersect_2nd[cubic_id] = answer
        return answer

    def get_motif_id_for_cubic_id(self,cubic_id):
        return cubic_id//2
            
    def get_bound_for_knot_x(self, knot_id):
        assert knot_id in self.additional_knot_ids
        return (self.epsilon_ratio,1-self.epsilon_ratio)
    
    def get_bound_for_knot_y_2nd(self, knot_id):
        if knot_id == self.n_cubic: # last knot
            if self.second_derivative_sign[knot_id-1] == 1:
                return (self.epsilon, np.inf)
            else:
                return (-np.inf, -self.epsilon)
        if self.second_derivative_sign[knot_id] == 1:
            return (self.epsilon, np.inf)
        else:
            return (-np.inf, -self.epsilon)
    
    def get_bounds(self):
        bounds = []
        for knot_id in self.additional_knot_ids:
            bounds.append(self.get_bound_for_knot_x(knot_id))
        for knot_id in self.knots_with_2nd_derivative_fixed:
            bounds.append(self.get_bound_for_knot_y_2nd(knot_id))
        for knot_id in self.fake_knots_y_undetermined:
            bounds.append(self.get_bound_for_knot_y_2nd(knot_id))
        return bounds
    
    def get_initial_guess(self,seed=42):
        # numpy random number generator
        gen = np.random.default_rng(seed=seed)

        initial_guess = []
        for knot_id in self.additional_knot_ids:
            initial_guess.append(0.5)
        for knot_id in self.knots_with_2nd_derivative_fixed:
            if knot_id == self.n_cubic: # last knot
                if self.second_derivative_sign[knot_id-1] == 1:
                    initial_guess.append(gen.uniform(self.epsilon,1))
                else:
                    initial_guess.append(-1*gen.uniform(self.epsilon,1))
            else:
                if self.second_derivative_sign[knot_id] == 1:
                    initial_guess.append(gen.uniform(self.epsilon,1))
                else:
                    initial_guess.append(-1*gen.uniform(self.epsilon,1))
        for knot_id in self.fake_knots_y_undetermined:
            if self.second_derivative_sign[knot_id] == 1:
                initial_guess.append(gen.uniform(self.epsilon,1))
            else:
                initial_guess.append(-1*gen.uniform(self.epsilon,1))
        return initial_guess
    
    def get_coefficients(self):
        coefficients = []
        for cubic_id in range(self.n_cubic):
            coefficients_piece = []
            coefficients_piece.append(self.get_slope_2nd(cubic_id)/6)
            coefficients_piece.append(self.get_intersect_2nd(cubic_id)/2)
            coefficients_piece.append(self.cs[cubic_id])
            coefficients_piece.append(self.ds[cubic_id])
            coefficients.append(coefficients_piece)
           
        return np.array(coefficients)
    
    def get_all_knots(self):
        return [self.get_knot_x(knot_id) for knot_id in range(len(self.knots))]

    def get_all_knots_y_2nd(self):
        return [self.get_knot_y_2nd(knot_id) for knot_id in range(len(self.knots))]
    
    def get_1st_derivative_sign_loss(self):
        def relu(x):
            return np.maximum(0,x)
        loss = 0
        for knot_id in range(len(self.knots)):
            if knot_id not in self.local_extrema_knots:
                if knot_id != 0:
                    if self.first_derivative_sign[knot_id-1] == 1:
                        value = -self.get_knot_y_1st(knot_id,side='left')
                    else:
                        value = self.get_knot_y_1st(knot_id,side='left')
                    loss += relu(value) * 1e6
                else:
                    if self.first_derivative_sign[knot_id] == 1:
                        value = -self.get_knot_y_1st(knot_id,side='right')
                    else:
                        value = self.get_knot_y_1st(knot_id,side='right')
                    loss += relu(value) * 1e6
        return loss

    def mse_loss(self):
        loss = 0
        for knot_id, knot in enumerate(self.knots):
            if knot.transition_point is not None:
                if knot_id != 0:
                    loss += (self.get_knot_y_0th(knot_id,side='left') - knot.y)**2
                if knot_id != len(self.knots) - 1:
                    loss += (self.get_knot_y_0th(knot_id,side='right') - knot.y)**2
        return loss
    
    def max_error(self):
        max_error = 0
        for knot_id, knot in enumerate(self.knots):
            if knot.transition_point is not None:
                if knot_id != 0:
                    error = np.abs(self.get_knot_y_0th(knot_id,side='left') - knot.y)
                    max_error = max(max_error,error)
                if knot_id != len(self.knots) - 1:
                    error = np.abs(self.get_knot_y_0th(knot_id,side='right') - knot.y)
                    max_error = max(max_error,error)
        return max_error
    def max_first_derivative_error(self):
        max_error = 0
        for knot_id, knot in enumerate(self.knots):
            if knot.transition_point in ['max','min']:
                if knot_id != 0:
                    error = np.abs(self.get_knot_y_1st(knot_id,side='left') - 0)
                    max_error = max(max_error,error)
                if knot_id != len(self.knots) - 1:
                    error = np.abs(self.get_knot_y_1st(knot_id,side='right') - 0)
                    max_error = max(max_error,error)
            elif knot.transition_point == 'start':
                error = np.abs(self.get_knot_y_1st(knot_id,side='right') - self.derivative_start)
                max_error = max(max_error,error)
            elif knot.transition_point == 'end':
                error = np.abs(self.get_knot_y_1st(knot_id,side='left') - self.derivative_end)
                max_error = max(max_error,error)
        return max_error

    

class PredictiveModel:

    def __init__(self, semantic_representation):
        self.semantic_representation = semantic_representation
        self.composition = semantic_representation.composition

        self.infinite_motif_classes = {
            '++f': PPF(),
            '+-p': PMP(),
            '+-h': PMH(),
            '-+f': MPF(),
            '-+h': MPH(),
            '--f': MMF()
        }

        self.compute_parameters()

       

    def predict(self,t):

        t_start = self.semantic_representation.t_range[0]

        y_pred = np.zeros_like(t)
        if self.semantic_representation.composition[0][2] == 'c':
            # there is at least one finite motif
            y_pred += np.where((t_start <= t) & (t<self.last_transition_point[0]), self.cubic_spline(t), 0)
        
        if self.semantic_representation.composition[-1][2] != 'c':
            y_pred += np.where(t >= self.last_transition_point[0], self.predict_infinite_motif(t), 0)
        else:
            y_pred += np.where(t == self.last_transition_point[0], self.cubic_spline(t), 0)
       
        
        return y_pred
    

    def normalize_trajectory(self, semantic_representation):

        # Copy the semantic representation
        semantic_representation = semantic_representation.copy()

        # Get the minimum and maximum values of the trajectory
        coordinates = semantic_representation.coordinates_finite_composition
        y_min = np.min(coordinates[:,1])
        y_max = np.max(coordinates[:,1])

        factor = 1/(y_max - y_min)

        # Scale so that y_max - y_min = 1
        coordinates[:,1] = (coordinates[:,1]) * factor

        semantic_representation.coordinates_finite_composition = coordinates

        # Scale the derivatives
        derivative_start = semantic_representation.derivative_start
        derivative_end = semantic_representation.derivative_end
        second_derivative_end = semantic_representation.second_derivative_end

        derivative_start = derivative_start * factor
        derivative_end = derivative_end * factor
        second_derivative_end = second_derivative_end * factor

        semantic_representation.derivative_start = derivative_start
        semantic_representation.derivative_end = derivative_end
        semantic_representation.second_derivative_end = second_derivative_end

        return semantic_representation, factor

    def compute_parameters(self):


        if self.semantic_representation.composition[0][2] == 'c':
        # there is at least one finite motif

            max_num_lbfgs_trials = 3
            max_num_powell_trials = 3
            default_seed = 2
            self.converged = False

            self.wrapped_spline = WrappedCubicSpline(self.semantic_representation)

            def optimize(trial, method):

                bounds = self.wrapped_spline.get_bounds()
                initial_guess = self.wrapped_spline.get_initial_guess(seed=default_seed+trial)

                def objective(z):
                    inner_loss = 0
                    self.wrapped_spline.set_parameters(z)
                
                    inner_loss += self.wrapped_spline.max_error()
                    inner_loss += self.wrapped_spline.get_1st_derivative_sign_loss()
                    if self.wrapped_spline.problem:
                        inner_loss += 1e6
                    return inner_loss
                
                if method == 'lbfgs':
                    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000, 'maxls': 100, 'ftol': 1e-9, 'gtol': 1e-9})
                elif method == 'powell':
                    result = minimize(objective, initial_guess, bounds=bounds, method='Powell', options={'maxiter': 1000, 'ftol': 1e-9})
                return result
            
            result = None
            best_result = None
            best_error = np.inf
            best_derivative_error = np.inf

             # print("Trying with Powell")
            for trial in range(max_num_powell_trials):
                result = optimize(trial, method='powell')

                error = self.wrapped_spline.max_error()
                derivative_error = self.wrapped_spline.max_first_derivative_error()

                if error < best_error:
                    best_result = result
                    best_error = error
                
                if derivative_error < best_derivative_error:
                    best_derivative_error = derivative_error

                if error < 1e-3 and derivative_error < 1e-3:
                    self.converged = True
                    # print(f'converged to {error}')
                    break
                else:
                    # print('powell')
                    # print(error)
                    # print(result.message)
                    pass

            if self.converged == False:
                
                for trial in range(max_num_lbfgs_trials):

                    result = optimize(trial, method='lbfgs')

                    error = self.wrapped_spline.max_error()
                    derivative_error = self.wrapped_spline.max_first_derivative_error()

                    if error < best_error:
                        best_result = result
                        best_error = error

                    if derivative_error < best_derivative_error:
                        best_derivative_error = derivative_error

                    if error < 1e-3 and derivative_error < 1e-3:
                        self.converged = True
                        # print(f'converged to {error}')
                        break
                    else:
                        # print('lbfgs')
                        # print(error)
                        # print(result.message)
                        pass
               

            if not self.converged:
                print("Smoothing did not converge for all seeds and methods")
                print(f"Best error: {best_error}")
                print(f"Best derivative error: {best_derivative_error}")
                print(self.semantic_representation)
            
            if self.converged:
                result = best_result

                self.wrapped_spline.set_parameters(result.x)
                self.all_knots = self.wrapped_spline.get_all_knots()
                self.all_coefficients = self.wrapped_spline.get_coefficients()
                self.cubic_spline = CubicSpline(self.wrapped_spline.get_all_knots(), self.wrapped_spline.get_coefficients().flatten())

                self.last_transition_point = (self.all_knots[-1],self.cubic_spline(self.all_knots[-1]))
                self.last_1st_derivative = self.cubic_spline.derivative(self.all_knots[-1])
                self.last_2nd_derivative = self.cubic_spline.second_derivative(self.all_knots[-1])
            else:
                self.last_transition_point = self.semantic_representation.coordinates_finite_composition[-1]
                self.last_1st_derivative = self.semantic_representation.derivative_end
                self.last_2nd_derivative = self.semantic_representation.second_derivative_end
        else:
            self.converged = True
            self.last_transition_point = self.semantic_representation.coordinates_finite_composition[-1]
            self.last_1st_derivative = self.semantic_representation.derivative_end
            self.last_2nd_derivative = self.semantic_representation.second_derivative_end
        

    def predict_infinite_motif(self,t):
        x0 = self.last_transition_point[0]
        y0 = self.last_transition_point[1]
        y1 = self.last_1st_derivative
        y2 = self.last_2nd_derivative
        # it's an infinite composition
        infinite_motif_properties = self.semantic_representation.properties_infinite_motif
        infinite_motif = self.composition[-1]
       
        motif_class = self.infinite_motif_classes[infinite_motif]
        t_to_use = np.where(t < x0, x0, t) # make sure we don't evaluate the infinite motif before the last transition point - this might cause errors
        return motif_class.evaluate_from_properties(t_to_use,infinite_motif_properties,x0,y0,y1,y2)
class CubicSpline:

    def __init__(self, internal_knots, coeffcients):
        """
            internal_knots: list of floats
            coeffcients: vector of size 4*len(internal_knots)
        """
        self.internal_knots = internal_knots
        self.coefficients = coeffcients

    def __call__(self,t):
        """
            t: sorted np.array of floats
        """
        t = np.atleast_1d(t)
        knots_x = self.internal_knots
        n_cubic = len(knots_x) - 1

        y = np.zeros_like(t)

        for i in range(n_cubic):
            evaluated_piece = self.coefficients[4*i]*t**3 + self.coefficients[4*i+1]*t**2 + self.coefficients[4*i+2]*t + self.coefficients[4*i+3]
            y += np.where((t >= knots_x[i]) & (t < knots_x[i+1]), evaluated_piece, 0)
        
        y += np.where(t == knots_x[-1], self.coefficients[-4]*t**3 + self.coefficients[-3]*t**2 + self.coefficients[-2]*t + self.coefficients[-1], 0)

        return y
    
    def derivative(self,t):
        """
            t: sorted np.array of floats
        """
        t = np.atleast_1d(t)
        knots_x = self.internal_knots
        n_cubic = len(knots_x) - 1

        y = np.zeros_like(t)
        cubic_counter = 0

        for i in range(len(t)):
            while t[i] > knots_x[cubic_counter+1]:
                cubic_counter += 1
                if cubic_counter == n_cubic:
                    raise ValueError('t is out of range')
            y[i] = 3*self.coefficients[4*cubic_counter]*t[i]**2 + 2*self.coefficients[4*cubic_counter+1]*t[i] + self.coefficients[4*cubic_counter+2]
        return y
    
    def second_derivative(self,t):
        """
            t: sorted np.array of floats
        """
        t = np.atleast_1d(t)
        knots_x = self.internal_knots
        n_cubic = len(knots_x) - 1

        y = np.zeros_like(t)
        cubic_counter = 0

        for i in range(len(t)):
            while t[i] > knots_x[cubic_counter+1]:
                cubic_counter += 1
                if cubic_counter == n_cubic:
                    raise ValueError('t is out of range')
            y[i] = 6*self.coefficients[4*cubic_counter]*t[i] + 2*self.coefficients[4*cubic_counter+1]
        return y

class ApproximatePredictiveModel:

    def __init__(self, semantic_representation):
        self.semantic_representation = semantic_representation
        self.composition = semantic_representation.composition

        if self.composition[-1][2] == 'c':
            self.infinite_composition = False
            self.infinite_motif = None
        else:
            self.infinite_composition = True
            self.infinite_motif = self.composition[-1]
        
        if self.infinite_composition:
            self.composition_finite_part = self.composition[:-1]
        else:
            self.composition_finite_part = self.composition

        self.infinite_motif_classes = {
            '++f': PPF(),
            '+-p': PMP(),
            '+-h': PMH(),
            '-+f': MPF(),
            '-+h': MPH(),
            '--f': MMF()
        }

    def type_of_transition_point(self, ind):
        composition = self.composition
        if ind == 0:
            return 'start'
        elif ind == len(composition):
            return 'end'
        else:
            if (composition[ind-1][:2] == "++") and (composition[ind][:2] == "+-"):
                return 'inflection'
            elif (composition[ind-1][:2] == "+-") and (composition[ind][:2] == "++"):
                return 'inflection'
            elif (composition[ind-1][:2] == "+-") and (composition[ind][:2] == "--"):
                return 'max'
            elif (composition[ind-1][:2] == "-+") and (composition[ind][:2] == "++"):
                return 'min'
            elif (composition[ind-1][:2] == "-+") and (composition[ind][:2] == "--"):
                return 'inflection'
            elif (composition[ind-1][:2] == "--") and (composition[ind][:2] == "-+"):
                return 'inflection'
            else:
                raise ValueError('Unknown transition point type')
            

    def get_coefficients_finite_composition(self):

        finite_coordinates = self.semantic_representation.coordinates_finite_composition
        
        coefficients_list = []

        for motif_index, motif in enumerate(self.composition_finite_part):
            coordinate_1 = finite_coordinates[motif_index,:]
            coordinate_2 = finite_coordinates[motif_index+1,:]

            A_row_0 = np.array([coordinate_1[0]**3, coordinate_1[0]**2, coordinate_1[0], 1.0])
            A_row_1 = np.array([coordinate_2[0]**3, coordinate_2[0]**2, coordinate_2[0], 1.0])
            b_0 = coordinate_1[1]
            b_1 = coordinate_2[1]

            type_1 = self.type_of_transition_point(motif_index)
            type_2 = self.type_of_transition_point(motif_index+1)
            if type_1 == 'max' or type_1 == 'min':
                A_row_2 = np.array([3*coordinate_1[0]**2, 2*coordinate_1[0], 1, 0])
                b_2 = 0
            elif type_1 == 'inflection':
                A_row_2 = np.array([6*coordinate_1[0], 2, 0, 0])
                b_2 = 0
            elif type_1 == 'start':
                A_row_2 = np.array([3*coordinate_1[0]**2, 2*coordinate_1[0], 1, 0])
                b_2 = self.semantic_representation.derivative_start
            
            if type_2 == 'max' or type_2 == 'min':
                A_row_3 = np.array([3*coordinate_2[0]**2, 2*coordinate_2[0], 1, 0])
                b_3 = 0
            elif type_2 == 'inflection':
                A_row_3 = np.array([6*coordinate_2[0], 2, 0, 0])
                b_3 = 0 
            elif type_2 == 'end':
                A_row_3 = np.array([3*coordinate_2[0]**2, 2*coordinate_2[0], 1, 0])
                b_3 = self.semantic_representation.derivative_end
                # We could have used the second derivative, it should not matter (in principle)
                # But in case something breakes I prefer to have matching first derivatives than second derivatives
           
            A = np.stack([A_row_0, A_row_1, A_row_2, A_row_3], axis=0)
            b = np.array([b_0, b_1, b_2, b_3])

            if np.abs(np.linalg.det(A)) < 1e-9:
                slope = (coordinate_2[1] - coordinate_1[1])/(coordinate_2[0]-coordinate_1[0])
                b = coordinate_1[1] - slope * coordinate_1[0]
                coefficients = np.array([0,0,slope,b])
            else:
                coefficients = np.linalg.solve(A, b)
            
            coefficients_list.append(coefficients)
        if len(coefficients_list) == 0:
            all_coefficients = np.zeros((0, 4))
        else:
            all_coefficients = np.stack(coefficients_list, axis=0) # shape (n_motifs, 4)
        return all_coefficients
    
    def evaluate_piece(self,finite_motif_coefficients, infinite_motif_properties, motif_index, t, last_transition_point=None,last_1st_derivative=None,last_2nd_derivative=None):
        if self.composition[motif_index][2] == 'c':
            # finite motif
            a = finite_motif_coefficients[motif_index,0]
            b = finite_motif_coefficients[motif_index,1]
            c = finite_motif_coefficients[motif_index,2]
            d = finite_motif_coefficients[motif_index,3]
            return a*t**3 + b*t**2 + c*t + d
        else:
            x0 = last_transition_point[0]
            y0 = last_transition_point[1]
            y1 = last_1st_derivative
            y2 = last_2nd_derivative
            # infinite motif
           
            motif_class = self.infinite_motif_classes[self.infinite_motif]
            T_to_use = np.where(t < x0, x0, t) # make sure we don't evaluate the infinite motif before the last transition point - this might cause errors
            return motif_class.evaluate_from_properties(T_to_use,infinite_motif_properties, x0, y0, y1, y2)
    def forward(self, t):
        """
        Forward pass of the model

        Args:
        """

        is_t_scalar = np.isscalar(t)

        t = np.atleast_1d(t)

        all_coefficients = self.get_coefficients_finite_composition()
        finite_coordinates = self.semantic_representation.coordinates_finite_composition
        knots = finite_coordinates[:,0]
        last_transition_point = finite_coordinates[-1,:]
        
        if self.infinite_composition:
            # add infinite knots
            knots = np.concatenate([knots, np.array([np.inf])])
            properties = self.semantic_representation.properties_infinite_motif
        else:
            properties = None

        y_pred = np.zeros(t.shape[0])

        for i in range(len(self.composition)):
            last_first_derivative = self.semantic_representation.derivative_end
            last_second_derivative = self.semantic_representation.second_derivative_end

            evaluated_piece = self.evaluate_piece(all_coefficients,properties,i,t,last_transition_point,last_first_derivative,last_second_derivative)

            y_pred += np.where((knots[i] <= t) & (t < knots[i+1]),evaluated_piece,0)
        
        if not self.infinite_composition:
            # Due the sharp inequalities earlier, we need to add the last piece separately
            a = all_coefficients[-1,0]
            b = all_coefficients[-1,1]
            c = all_coefficients[-1,2]
            d = all_coefficients[-1,3]
            y_pred += np.where(t == knots[-1],a*t**3 + b*t**2 + c*t + d,0)
            # possibly add values beyond last t based on some condition, you can use first or second derivite information
        
        if is_t_scalar:
            return y_pred[0]
        else:
            return y_pred