import numpy as np
from semantic_odes.infinite_motifs import *
from scipy.optimize import minimize

MIN_TRANSITION_POINT_SEP = 1e-2
MIN_PROPERTY_VALUE = 1e-3
MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT = 0.2

def softplus(x):
    # Use piecewise to handle different ranges of x
    mask = x > 20
    res = np.zeros_like(x)
    res[mask] = x[mask]
    res[~mask] = np.log1p(np.exp(x[~mask]))
    return res

def sigmoid(x, threshold=20.0):
    maskg20 = x > threshold
    masklm20 = x < -threshold
    res = np.zeros_like(x)
    res[maskg20] = 1.0
    res[masklm20] = 0.0
    res[~maskg20 & ~masklm20] = 1 / (1 + np.exp(-x[~maskg20 & ~masklm20]))
    return res


def softmax(x, temperature=1.0):
    """
    Compute the softmax of vector x with a temperature parameter.
    
    Parameters:
    x (numpy.ndarray): Input data.
    temperature (float): Temperature parameter for scaling.
    
    Returns:
    numpy.ndarray: Softmax probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    
    x = np.array(x) / temperature
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    return e_x / sum_e_x


        
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

        # self.weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_coordinates))
        # self.horizontal_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, len(self.composition_finite_part)))
        # self.vertical_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, len(self.composition_finite_part)))

        # Special properties of the infinite motif
        if self.infinite_composition:
            num_properties = self.number_of_properties_for_infinite_motif()
            # self.infinite_motif_properties_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, num_properties))
            if len(self.composition) != 1:
                # self.infinite_motif_start_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
                pass

        if self.composition[0][2] == 'c':
            # if the first motif is finite we have an additional degree of freedom
            # for the first derivative at the start
            # self.first_derivative_at_start = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
            self.specified_first_derivative_at_start = True
        else:
            self.specified_first_derivative_at_start = False

        if self.composition[-1][2] == 'c' and len(self.composition) != 1:
            # if the last motif is finite (and it's not the only motif) we have an additional degree of freedom
            # for the first derivative at the end
            # with infinite composition, this is not needed as the last finite transition point is always an extremum or transition point
            # self.first_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
            self.specified_first_derivative_at_end = True
            self.specified_second_derivative_at_end = False
        elif self.infinite_composition and len(self.composition) == 1:
            # if the last motif is infinite and it's the only motif, we have an additional degree of freedom
            # for the first derivative at the end
            self.specified_first_derivative_at_end = True
            second_derivative_vanishes = self.infinite_motif_classes[self.infinite_motif].second_derivative_vanishes()
            if not second_derivative_vanishes:
                self.specified_second_derivative_at_end = True
            else:
                self.specified_second_derivative_at_end = False
        else:
            self.specified_first_derivative_at_end = False
            self.specified_second_derivative_at_end = False

        self.arguments = {}


    def set_arguments(self, arguments):
        self.arguments = arguments




    def number_of_properties_for_infinite_motif(self):
        if not self.infinite_composition:
            return 0
        return self.infinite_motif_classes[self.infinite_motif].num_network_properties()
        
            
    def extract_first_derivative_at_start(self):
        if self.specified_first_derivative_at_start:

            finite_coordinates = self.extract_coordinates_finite_composition()

            first_derivative_at_start = self.arguments['first_derivative_at_start']

            motif_index = 0
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            calculated_first_derivative_ratio = sigmoid(first_derivative_at_start)
            slope_min, slope_max = self.get_first_derivative_range(0, coordinate_1, coordinate_2, "left")
            return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
                
        else:
            return None
        
    def extract_first_derivative_at_end(self):
        raise NotImplementedError
        if self.specified_first_derivative_at_end:

            finite_coordinates = self.extract_coordinates_finite_composition()

            first_derivative_at_end = self.arguments['first_derivative_at_end']

            motif_index = len(self.composition_finite_part) - 1
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            calculated_first_derivative_ratio = sigmoid(first_derivative_at_end)
            slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "right")
            return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
    
    def extract_properties_infinite_motif(self):
        raise NotImplementedError
        infinite_properties = self.arguments['infinite_properties']

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition()
        knots = finite_coordinates[:,0]
        last_transition_point = finite_coordinates[-1,:]

        if self.infinite_composition:
            # add infinite knots
            knots = np.concatenate([knots, np.array([np.inf])],axis=0)
            properties = softplus(infinite_properties) + MIN_PROPERTY_VALUE
        else:
            properties = None

        i = len(self.composition) - 1

        if self.composition[i][2] != 'c' and i > 0:
            last_first_derivative = 3*all_coefficients[i-1,0] * last_transition_point[0]**2 + 2 * all_coefficients[i-1,1] * last_transition_point[0] + all_coefficients[i-1,2]
            last_second_derivative = 6*all_coefficients[i-1,0] * last_transition_point[0] + 2 * all_coefficients[i-1,1]
        else:
            last_first_derivative = None
            last_second_derivative = None

        x0 = last_transition_point[0]
        y0 = last_transition_point[1]
       
        if len(self.composition) == 1:
            # There is only an infinite motif
        #    motif_class = self.infinite_motif_single_classes[self.infinite_motif]
           return motif_class.extract_properties_from_network(properties.reshape((1,-1)), x0, y0, last_first_derivative, last_second_derivative).flatten()
        else:
            # There is a previous motif
            motif_class = self.infinite_motif_classes[self.infinite_motif]
            return motif_class.extract_properties_from_network(properties.reshape((1,-1)), x0, y0, last_first_derivative, last_second_derivative).flatten()
      
    
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
                t_last_finite_transition_point = self.t_range[0] + MIN_TRANSITION_POINT_SEP + sigmoid(t_last_finite_transition_point) * (1-MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT) * empirical_range
            else:
                t_last_finite_transition_point = self.t_range[0]
        else:
            t_last_finite_transition_point = self.t_range[1]



        all_coordinate_values = np.zeros((len(self.composition_finite_part)+1,2))

        # pass horizontal values through a softmax function, and scale them by the range of the time points
        scale = t_last_finite_transition_point - self.t_range[0]

        temp = 1.0
        trans_horizontal_values = softmax(horizontal_values, temperature=temp) * scale

        for _ in range(10):
            if np.min(trans_horizontal_values) >= MIN_TRANSITION_POINT_SEP:
                break
            temp *= 2
            trans_horizontal_values = softmax(horizontal_values, temperature=temp) * scale

        horizontal_values = trans_horizontal_values


        # make sure vertical values are positive
        vertical_values = softplus(vertical_values)
        
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

    def get_first_derivative_range(self, motif_index, point1, point2, which_point):
        if np.isclose(point2[0], point1[0]):
            print("exception")
            print(point1, point2)
            
        slope = (point2[1] - point1[1])/(point2[0] - point1[0])
        motif = self.composition[motif_index]

        if which_point == 'left':
            if motif == '++c':
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return 0, slope
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return 0, slope
            elif motif == "+-c":
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return slope, 2 * slope
                elif self.type_of_transition_point(motif_index+1) == 'max':
                    return 1.5 * slope, 3 * slope
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return slope, 3 * slope
            elif motif == "-+c":
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return slope, 2 * slope
                elif self.type_of_transition_point(motif_index+1) == 'min':
                    return 1.5 * slope, 3 * slope
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return slope, 3 * slope
            elif motif == "--c":
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return 0, slope
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return 0, slope
        
        elif which_point == 'right':
            if motif == '++c':
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return slope, 3*slope
                elif self.type_of_transition_point(motif_index) == 'min':
                    return 1.5 * slope, 3 * slope
            elif motif == "+-c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return 0, slope
            elif motif == "-+c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return 0, slope
            elif motif == "--c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return slope, 3 * slope
                elif self.type_of_transition_point(motif_index) == 'max':
                    return 1.5 * slope, 3 * slope
                
        # in general left of -a and right of +a are the same and vice versa.
                
    def get_coefficients_and_coordinates_finite_composition(self):

        first_derivative_at_start = self.arguments['first_derivative_at_start']
        first_derivative_at_end = self.arguments['first_derivative_at_end']

        finite_coordinates = self.extract_coordinates_finite_composition() # shape (n_all_coordinates, 2)
        
        b = np.zeros(3)
        coefficients_list = []

        for motif_index, motif in enumerate(self.composition_finite_part):
            coordinate_1 = finite_coordinates[motif_index,:]
            coordinate_2 = finite_coordinates[motif_index+1,:]

            A_row_0 = np.array([coordinate_1[0]**3, coordinate_1[0]**2, coordinate_1[0], 1])
            A_row_1 = np.array([coordinate_2[0]**3, coordinate_2[0]**2, coordinate_2[0], 1])
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
            elif type_1 == 'start' and self.specified_first_derivative_at_start:
                A_row_2 = np.array([3*coordinate_1[0]**2, 2*coordinate_1[0], 1, 0])
                calculated_first_derivative_ratio = sigmoid(first_derivative_at_start)
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "left")
                b_2 = slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
                if type_2 == 'end':
                    # in that case we reduce the cubic to a quadratic
                    A_row_3 = np.array([1, 0, 0, 0])
                    b_3 = 0
            if type_2 == 'max' or type_2 == 'min':
                A_row_3 = np.array([3*coordinate_2[0]**2, 2*coordinate_2[0], 1, 0])
                b_3 = 0
            elif type_2 == 'inflection':
                A_row_3 = np.array([6*coordinate_2[0], 2, 0, 0])
                b_3 = 0
            elif (type_2 == 'end' and self.specified_first_derivative_at_end) and type_1 != 'start':
                A_row_3 = np.array([3*coordinate_2[0]**2, 2*coordinate_2[0], 1, 0])
                calculated_first_derivative_ratio = sigmoid(first_derivative_at_end)
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "right")
                b_3 = slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)

            A = np.stack([A_row_0, A_row_1, A_row_2, A_row_3], axis=0)
            b = np.stack([b_0, b_1, b_2, b_3], axis=0)

            if np.abs(np.linalg.det(A)) < 1e-9:
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
            properties = softplus(infinite_properties) + MIN_PROPERTY_VALUE
        else:
            properties = None

        if T is None:
            T = self.t
        
        Y_pred = np.zeros(len(T))

        for i in range(len(self.composition)):
            if self.composition[i][2] != 'c' and i > 0:
                last_first_derivative = 3*all_coefficients[i-1,0] * last_transition_point[0]**2 + 2 * all_coefficients[i-1,1] * last_transition_point[0] + all_coefficients[i-1,2]
                last_second_derivative = 6*all_coefficients[i-1,0] * last_transition_point[0] + 2 * all_coefficients[i-1,1]
            elif self.composition[i][2] != 'c' and i == 0:
                assert self.specified_first_derivative_at_end
                sign = 1 if self.composition[-1][0] == '+' else -1
                last_first_derivative = sign * softplus(self.arguments['first_derivative_at_end'])
                if self.specified_second_derivative_at_end:
                    sign = 1 if self.composition[-1][1] == '+' else -1
                    last_second_derivative = sign * softplus(self.arguments['second_derivative_at_end'])
                else:
                    last_second_derivative = np.zeros_like(last_first_derivative)
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

    if model.specified_first_derivative_at_start:
        n_parameters['first_derivative_at_start'] = 1
    else:
        n_parameters['first_derivative_at_start'] = 0
    
    if model.specified_first_derivative_at_end:
        n_parameters['first_derivative_at_end'] = 1
    else:
        n_parameters['first_derivative_at_end'] = 0
    
    if model.specified_second_derivative_at_end:
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



def calculate_loss_3_samples(composition, t_range, before, middle, after, seed=0, train_on_all_data=False):

    before_x0, before_t, before_y = before
    middle_x0, middle_t, middle_y = middle
    after_x0, after_t, after_y = after

    gen = np.random.default_rng(seed=seed)

    if train_on_all_data:
        before_t_train = before_t
        before_y_train = before_y
        middle_t_train = middle_t
        middle_y_train = middle_y
        after_t_train = after_t
        after_y_train = after_y
    else:
        before_train_indices = np.arange(int(0.8*len(before_t)))
        middle_train_indices = np.arange(int(0.8*len(middle_t)))
        after_train_indices = np.arange(int(0.8*len(after_t)))

        before_t_train = before_t[before_train_indices]
        before_y_train = before_y[before_train_indices]
        middle_t_train = middle_t[middle_train_indices]
        middle_y_train = middle_y[middle_train_indices]
        after_t_train = after_t[after_train_indices]
        after_y_train = after_y[after_train_indices]

        before_t_test = before_t[before_train_indices]
        before_y_test = before_y[before_train_indices]
        middle_t_test = middle_t[middle_train_indices]
        middle_y_test = middle_y[middle_train_indices]
        after_t_test = after_t[after_train_indices]
        after_y_test = after_y[after_train_indices]
    

    model_before = CubicModelNumpy(composition, t_range, before_x0, before_t, before_y)
    model_middle = CubicModelNumpy(composition, t_range, middle_x0, middle_t, middle_y)
    model_after = CubicModelNumpy(composition, t_range, after_x0, after_t, after_y)

    dictionary, n_all_parameters = create_dictionary_for_a_model(model_middle)

    initial_guess = gen.normal(size=n_all_parameters * 2)

    def loss_function(weights_and_biases):

        weights = weights_and_biases[:n_all_parameters]
        biases = weights_and_biases[n_all_parameters:]

        before_parameters = before_x0 * weights + biases
        middle_parameters = middle_x0 * weights + biases
        after_parameters = after_x0 * weights + biases

        before_arguments = from_parameters_to_arguments(before_parameters, dictionary)
        middle_arguments = from_parameters_to_arguments(middle_parameters, dictionary)
        after_arguments = from_parameters_to_arguments(after_parameters, dictionary)

        model_before.set_arguments(before_arguments)
        model_middle.set_arguments(middle_arguments)
        model_after.set_arguments(after_arguments)

        before_y_pred = model_before.forward(before_t_train)
        middle_y_pred = model_middle.forward(middle_t_train)
        after_y_pred = model_after.forward(after_t_train)

        before_loss = np.mean((before_y_pred - before_y_train)**2)
        middle_loss = np.mean((middle_y_pred - middle_y_train)**2)
        after_loss = np.mean((after_y_pred - after_y_train)**2)


        return before_loss + 2*middle_loss + after_loss
    
    result = minimize(loss_function, initial_guess, method='L-BFGS-B')

    weights = result.x[:n_all_parameters]
    biases = result.x[n_all_parameters:]

    middle_parameters = middle_x0 * weights + biases
    middle_arguments = from_parameters_to_arguments(middle_parameters, dictionary)
    model_middle.set_arguments(middle_arguments)

    if train_on_all_data:
        middle_y_pred_test = model_middle.forward(middle_t)
        final_loss = np.mean((middle_y_pred_test - middle_y)**2)
    else:
        middle_y_pred_test = model_middle.forward(middle_t_test)
        final_loss = np.mean((middle_y_pred_test - middle_y_test)**2)
    
    return final_loss, model_middle


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

    



    


