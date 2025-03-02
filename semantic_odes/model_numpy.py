import numpy as np
from semantic_odes.infinite_motifs import *
from scipy.optimize import minimize
from semantic_odes import utils

MIN_TRANSITION_POINT_SEP = 1e-2
MIN_PROPERTY_VALUE = 1e-3
MIN_VERTICAL_SEPARATION = 1e-3
MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT = 0.2

        
class CubicModelNumpy:
    def __init__(self, composition, t_range, x0, t, y):

        self.infinite_motif_classes = {
            '++f': PPF(),
            '+-p': PMP(),
            '+-h': PMH(),
            '-+f': MPF(),
            '-+h': MPH(),
            '--f': MMF()
        }
        
        self.t_range = t_range
        self.composition = composition
        self.n_coordinates = 2*len(self.composition)
        self.x0 = x0
        self.y = y
        self.t = t


        if utils.is_unbounded_composition(self.composition):
            # it is an infinite composition
            self.composition_finite_part = self.composition[:-1]
            self.infinite_motif = self.composition[-1]
            self.infinite_composition = True
        else:
            # it is a finite composition
            self.composition_finite_part = self.composition
            self.infinite_motif = None
            self.infinite_composition = False

        self.first_derivative_at_start_status = utils.get_first_derivative_at_start_status(self.composition)
        self.first_derivative_at_end_status = utils.get_first_derivative_at_end_status(self.composition)
        self.second_derivative_at_end_status = utils.get_second_derivative_at_end_status(self.composition)

        self.arguments = {}


    def set_arguments(self, arguments):
        self.arguments = arguments


    def number_of_properties_for_infinite_motif(self):
        if not self.infinite_composition:
            return 0
        return self.infinite_motif_classes[self.infinite_motif].num_network_properties()
        
            
    def extract_first_derivative_at_start(self):
        if self.first_derivative_at_start_status == 'weights':

            finite_coordinates = self.extract_coordinates_finite_composition()

            first_derivative_at_start = self.arguments['first_derivative_at_start']

            motif_index = 0
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            calculated_first_derivative_ratio = utils.sigmoid_np(first_derivative_at_start)
            slope_min, slope_max = self.get_first_derivative_range(0, coordinate_1, coordinate_2, "left")
            return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
                
        else:
            return None
    
    def extract_coordinates_finite_composition(self):
        x0 = self.x0
        horizontal_values = self.arguments['horizontal_values']
        vertical_values = self.arguments['vertical_values']
        t_last_finite_transition_point = self.arguments['t_last_finite_transition_point']

        if self.infinite_composition:
            if len(self.composition) > 1:
                # When fitting an individual sample, it makes sense to limit the position of the last finite transition point
                # to the range of the time points
                empirical_t_max = np.max(self.t)
                empirical_range = empirical_t_max - self.t_range[0]
                t_last_finite_transition_point = self.t_range[0] + MIN_TRANSITION_POINT_SEP + utils.sigmoid_np(t_last_finite_transition_point) * (1-MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT) * empirical_range
            else:
                t_last_finite_transition_point = self.t_range[0]
        else:
            t_last_finite_transition_point = self.t_range[1]

        all_coordinate_values = np.zeros((len(self.composition_finite_part)+1,2))

        # if len(self.composition) > 1:
        #     # pass horizontal values through a softmax function, and scale them by the range of the time points
        #     scale = t_last_finite_transition_point - self.t_range[0]

        #     trans_horizontal_values = utils.softmax_np(horizontal_values) * scale
        #     good_indices = np.arange(len(trans_horizontal_values))
        #     bad_indices = []

        #     while np.min(trans_horizontal_values) < MIN_TRANSITION_POINT_SEP:
        #         scale -= MIN_TRANSITION_POINT_SEP
        #         i = np.argmin(trans_horizontal_values)
        #         good_indices = good_indices[good_indices != i]
        #         bad_indices.append(i)
        #         new_horizontal_values = utils.softmax_np(horizontal_values[good_indices]) * scale
        #         trans_horizontal_values = np.zeros_like(horizontal_values)
        #         trans_horizontal_values[good_indices] = new_horizontal_values
        #         trans_horizontal_values[bad_indices] = MIN_TRANSITION_POINT_SEP

        #         horizontal_values = trans_horizontal_values

        # pass horizontal values through a softmax function, and scale them by the range of the time points
        scale = t_last_finite_transition_point - self.t_range[0]

        temp = 1.0
        trans_horizontal_values = utils.softmax_np(horizontal_values, temperature=temp) * scale

        for _ in range(10):
            if np.min(trans_horizontal_values) >= MIN_TRANSITION_POINT_SEP:
                break
            temp *= 2
            trans_horizontal_values = utils.softmax_np(horizontal_values, temperature=temp) * scale

        horizontal_values = trans_horizontal_values

        # make sure vertical values are positive
        vertical_values = utils.softplus_np(vertical_values) # + MIN_VERTICAL_SEPARATION
        
        # j = 0
        all_coordinate_values[0,0] = self.t_range[0]
        all_coordinate_values[0,1] = x0

        for j in range(1, len(self.composition_finite_part)+1):
            sign = 1 if self.composition_finite_part[j-1][0] == '+' else -1
            all_coordinate_values[j,0] = horizontal_values[j-1] + all_coordinate_values[j-1,0]
            all_coordinate_values[j,1] = sign*vertical_values[j-1] + all_coordinate_values[j-1,1]

        # Force the last point to be the end of the time range
        all_coordinate_values[-1,0] = t_last_finite_transition_point
        return all_coordinate_values

    def type_of_transition_point(self, ind):
        """
        Determine the type of the transition point

        Args:
            ind: int, the index of the transition point
        
        Returns:
            str, the type of the transition point
        """
        composition = self.composition
        return utils.type_of_transition_point(composition, ind)

    def get_first_derivative_range(self, motif_index, point1, point2, which_point):
        """
        Get the range of the first derivative at a transition point

        Args:
            motif_index: int, the index of the motif
            point1: numpy array of shape (2,) with the x and y coordinates of the first point
            point2: numpy array of shape (2,) with the x and y coordinates of the second point
            which_point: str, either "left" or "right" to specify which point to consider
        
        Returns:
            tuple of two floats, the minimum and maximum values of the first derivative
        """
        if np.isclose(point2[0], point1[0]):
            print("The two transition points are too close")
            print(point1, point2)
            
        slope = (point2[1] - point1[1])/(point2[0] - point1[0])

        coefficients = utils.get_first_derivative_range_coefficients(self.composition, motif_index, which_point)
        return coefficients[0] * slope, coefficients[1] * slope
    
    def _create_row(self, point, order):
        """
        Create a row of the matrix A for the system of equations to find the coefficients of the cubic polynomial

        Args:
            point: numpy array of shape (2,) with the x and y coordinates of the point
            order: int, the order of the derivative to be used in the row
        
        Returns:
            numpy array of shape (4,) with the coefficients of the row
        """
        if order == 0:
            return np.array([point[0]**3, point[0]**2, point[0], 1])
        elif order == 1:
            return np.array([3*point[0]**2, 2*point[0], 1, 0])
        elif order == 2:
            return np.array([6*point[0], 2, 0, 0])
        else:
            raise ValueError("Invalid order")
                
    def get_coefficients_and_coordinates_finite_composition(self):

        first_derivative_at_start = self.arguments['first_derivative_at_start']
        first_derivative_at_end = self.arguments['first_derivative_at_end']

        finite_coordinates = self.extract_coordinates_finite_composition() # shape (n_all_coordinates, 2)
        
        b = np.zeros(3)
        coefficients_list = []

        for motif_index, motif in enumerate(self.composition_finite_part):
            coordinate_1 = finite_coordinates[motif_index,:]
            coordinate_2 = finite_coordinates[motif_index+1,:]

            A_row_0 = self._create_row(coordinate_1, 0)
            A_row_1 = self._create_row(coordinate_2, 0)
            b_0 = coordinate_1[1]
            b_1 = coordinate_2[1]

            type_1 = self.type_of_transition_point(motif_index)
            type_2 = self.type_of_transition_point(motif_index+1)
            if type_1 == 'max' or type_1 == 'min':
                A_row_2 = self._create_row(coordinate_1, 1)
                b_2 = 0
            elif type_1 == 'inflection':
                A_row_2 = self._create_row(coordinate_1, 2)
                b_2 = 0
            elif type_1 == 'start' and self.first_derivative_at_start_status == 'weights':
                A_row_2 = self._create_row(coordinate_1, 1)
                calculated_first_derivative_ratio = utils.sigmoid_np(first_derivative_at_start)
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "left")
                b_2 = slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
                if type_2 == 'end':
                    # in that case we reduce the cubic to a quadratic
                    A_row_3 = np.array([1, 0, 0, 0])
                    b_3 = 0
            if type_2 == 'max' or type_2 == 'min':
                A_row_3 = self._create_row(coordinate_2, 1)
                b_3 = 0
            elif type_2 == 'inflection':
                # A_row_3 = np.array([6*coordinate_2[0], 2, 0, 0])
                A_row_3 = self._create_row(coordinate_2, 2)
                b_3 = 0
            elif (type_2 == 'end' and self.first_derivative_at_end_status == 'weights') and type_1 != 'start':
                A_row_3 = self._create_row(coordinate_2, 1)
                calculated_first_derivative_ratio = utils.sigmoid_np(first_derivative_at_end)
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "right")
                b_3 = slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)

            A = np.stack([A_row_0, A_row_1, A_row_2, A_row_3], axis=0)
            b = np.stack([b_0, b_1, b_2, b_3], axis=0)

            if np.abs(np.linalg.det(A)) < 1e-9 or np.abs(np.linalg.det(A)) > 1e9:
                # print("exception")
                # just connect with a line
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
        return all_coefficients, finite_coordinates

    def evaluate_piece(self,finite_motif_coefficients, infinite_motif_properties, motif_index, T, last_transition_point=None,last_1st_derivative=None,last_2nd_derivative=None):
        if self.composition[motif_index][2] == 'c':
            # finite motif
            a = finite_motif_coefficients[motif_index,0]
            b = finite_motif_coefficients[motif_index,1]
            c = finite_motif_coefficients[motif_index,2]
            d = finite_motif_coefficients[motif_index,3]
            return a*T**3 + b*T**2 + c*T + d
        else:
            x0 = last_transition_point[0]
            y0 = last_transition_point[1]
            y1 = last_1st_derivative
            y2 = last_2nd_derivative
            # infinite motif
            motif_class = self.infinite_motif_classes[self.infinite_motif]
            T_to_use = np.where(T < x0, x0, T) # make sure we don't evaluate the infinite motif before the last transition point - this might cause errors
            return motif_class.evaluate_from_network(T_to_use,infinite_motif_properties.reshape((1,-1)), x0, y0, y1, y2).flatten()
    def forward(self, T=None):
        """
        Forward pass of the model

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)
        T: input tensor of shape (batch_size, n_time_points)
        """

        infinite_properties = self.arguments['infinite_properties']


        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition()
        knots = finite_coordinates[:,0]
        last_transition_point = finite_coordinates[-1,:]

        if self.infinite_composition:
            # add infinite knots
            knots = np.concatenate([knots, np.array([np.inf])],axis=0)
            properties = utils.softplus_np(infinite_properties) + MIN_PROPERTY_VALUE
        else:
            properties = None

        if T is None:
            T = self.t
        
        Y_pred = np.zeros(len(T))

        for i in range(len(self.composition)):

            if utils.is_unbounded_motif(self.composition[i]):
                # We need to specify the derivatives at end

                last_first_derivative_status = utils.get_first_derivative_at_end_status(self.composition)
                last_second_derivative_status = utils.get_second_derivative_at_end_status(self.composition)

                if last_first_derivative_status == "zero":
                    last_first_derivative = 0
                elif last_first_derivative_status == "cubic":
                    last_first_derivative = utils.evaluate_cubic(all_coefficients[i-1], last_transition_point[0], derivative_order=1)
                elif last_first_derivative_status == "weights":
                    sign = 1 if self.composition[-1][0] == '+' else -1
                    last_first_derivative = sign * utils.softplus_np(self.arguments['first_derivative_at_end'])
                else:
                    raise ValueError("Unknown first derivative status")
                
                if last_second_derivative_status == "zero":
                    last_second_derivative = 0
                elif last_second_derivative_status == "cubic":
                    last_second_derivative = utils.evaluate_cubic(all_coefficients[i-1], last_transition_point[0], derivative_order=2)
                elif last_second_derivative_status == "weights":
                    sign = 1 if self.composition[-1][1] == '+' else -1
                    last_second_derivative = sign * utils.softplus_np(self.arguments['second_derivative_at_end'])
                else:
                    raise ValueError("Unknown second derivative status")

            else:
                last_first_derivative = None
                last_second_derivative = None

            evaluated_piece = self.evaluate_piece(all_coefficients,properties,i,T,last_transition_point,last_first_derivative,last_second_derivative)

            Y_pred += np.where((knots[i] <= T) & (T < knots[i+1]),evaluated_piece,0)
        
        if not self.infinite_composition:
            # Due the sharp inequalities earlier, we need to add the last piece separately
            a = all_coefficients[-1,0]
            b = all_coefficients[-1,1]
            c = all_coefficients[-1,2]
            d = all_coefficients[-1,3]
            Y_pred += np.where(T == knots[-1],a*T**3 + b*T**2 + c*T + d,0)
            # possibly add values beyond last t based on some condition, you can use first or second derivite information
        
        return Y_pred
    

def from_parameters_to_arguments(parameters, dictionary):

    arguments = {}

    for key, value in dictionary.items():
        if key in ['horizontal_values', 'vertical_values']:
            if value is not None:
                arguments[key] = parameters[value[0]:value[1]]
            else:
                arguments[key] = np.zeros(1)
        elif key == 'infinite_properties':
            if value is not None:
                arguments[key] = parameters[value[0]:value[1]]
            else:
                arguments[key] = None
        else:
            if value is not None:
                arguments[key] = parameters[value[0]]
            else:
                arguments[key] = None
    
    return arguments

def create_dictionary_for_a_model(model):
    n_parameters = {}
    
    n_parameters['horizontal_values'] = len(model.composition_finite_part)
    n_parameters['vertical_values'] = len(model.composition_finite_part)
    n_parameters['infinite_properties'] = model.number_of_properties_for_infinite_motif()


    n_parameters['t_last_finite_transition_point'] = 0
    if model.infinite_composition:
        if len(model.composition) != 1:
            n_parameters['t_last_finite_transition_point'] = 1

    if model.first_derivative_at_start_status == 'weights':
        n_parameters['first_derivative_at_start'] = 1
    else:
        n_parameters['first_derivative_at_start'] = 0
    
    if model.first_derivative_at_end_status == 'weights':
        n_parameters['first_derivative_at_end'] = 1
    else:
        n_parameters['first_derivative_at_end'] = 0
    
    if model.second_derivative_at_end_status == 'weights':
        n_parameters['second_derivative_at_end'] = 1
    else:
        n_parameters['second_derivative_at_end'] = 0

    dictionary = {}
    current_index = 0
    for key, value in n_parameters.items():
        if value != 0:
            dictionary[key] = (current_index, current_index+value)
        else:
            dictionary[key] = None
        current_index += value

    n_all_parameters = current_index

    return dictionary, n_all_parameters



def calculate_loss(composition, t_range, x0, t, y,seed=0, train_on_all_data=False, evaluate_on_all_data=True):

    # train test split of t and y
    gen = np.random.default_rng(seed=seed)
    # train_indices = gen.choice(len(t), int(0.8*len(t)), replace=False)
    # test_indices = np.array([i for i in range(len(t)) if i not in train_indices])
    if train_on_all_data:
        t_train = t
        y_train = y
    else:
        train_indices = np.arange(int(0.8*len(t)))
        test_indices = np.arange(int(0.8*len(t)), len(t))
        t_train = t[train_indices]
        y_train = y[train_indices]
        t_test = t[test_indices]
        y_test = y[test_indices]

    model = CubicModelNumpy(composition, t_range, x0, t, y)

    dictionary, n_all_parameters = create_dictionary_for_a_model(model)
    
    initial_guess = gen.normal(size=n_all_parameters)
    # initial_guess = np.zeros(n_all_parameters)

    def loss_function(parameters):

        arguments = from_parameters_to_arguments(parameters, dictionary)
        y_pred = model.set_arguments(arguments)
        y_pred = model.forward(t_train)
        return np.mean((y_pred - y_train)**2)
    
    # Use scipy to minimize the loss function
    result = minimize(loss_function, initial_guess, method='L-BFGS-B')
    
    # Return the loss
    model.set_arguments(from_parameters_to_arguments(result.x, dictionary))

    normalize = False
    if evaluate_on_all_data:
        y_pred_test = model.forward(t)
        if normalize:
            y_range = np.max(y) - np.min(y)
            final_loss = np.sqrt(np.mean((y_pred_test - y)**2))/y_range
        else:
            final_loss = np.sqrt(np.mean((y_pred_test - y)**2))
    else:
        y_pred_test = model.forward(t_test)
        if normalize:
            y_range = np.max(y_test) - np.min(y_test)
            final_loss = np.sqrt(np.mean((y_pred_test - y_test)**2))/y_range
        else:
            final_loss = np.sqrt(np.mean((y_pred_test - y_test)**2))
    
    return final_loss, model

    



    


