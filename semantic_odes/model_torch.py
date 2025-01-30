import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline
import numpy as np
from semantic_odes.infinite_motifs import *

MIN_TRANSITION_POINT_SEP = 1e-1
# MIN_PROPERTY_VALUE = 1e-3
MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT = 1e-1


class CubicModel(torch.nn.Module):
    def __init__(self, config, fixed_infinite_properties={}):
        super(CubicModel, self).__init__()

        self.infinite_motif_classes = {
            '++f': PPF(),
            '+-p': PMP(),
            '+-h': PMH(),
            '-+f': MPF(),
            '-+h': MPH(),
            '--f': MMF()
        }
        
        self.config = config
        
        self.composition = config['composition']
        self.n_basis_functions = config['n_basis_functions']
        self.seed = config['seed']
        self.n_coordinates = 2*len(self.composition)

        self.dis_loss_coeff_1 = config['dis_loss_coeff_1']
        self.dis_loss_coeff_2 = config['dis_loss_coeff_2']

        self.fixed_infinite_properties = fixed_infinite_properties

        if config['device'] == 'gpu' or config['device'] == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if config['t_range'][1] is None:
            self.t_range = torch.Tensor([config['t_range'][0],np.inf]).to(self.device)
        else:
            self.t_range = torch.Tensor([config['t_range'][0],config['t_range'][1]]).to(self.device)
        torch.manual_seed(self.seed)
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

        self.scalers = {}

        # self.weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, self.n_coordinates))
        self.horizontal_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, len(self.composition_finite_part)))
        self.vertical_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, len(self.composition_finite_part)))

        self.scalers['vertical'] = torch.nn.Parameter(torch.randn(len(self.composition_finite_part)))

        # Special properties of the infinite motif
        if self.infinite_composition:
            num_properties = self.number_of_properties_for_infinite_motif()
            self.infinite_motif_properties_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, num_properties))
            self.scalers['infinite_motif_properties'] = torch.nn.Parameter(torch.randn(num_properties))
            if len(self.composition) != 1:
                self.infinite_motif_start_weights = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
                self.scalers['infinite_motif_start'] = torch.nn.Parameter(torch.randn(1))

        if self.composition[0][2] == 'c':
            # if the first motif is finite we have an additional degree of freedom
            # for the first derivative at the start
            self.first_derivative_at_start = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))

            self.specified_first_derivative_at_start = True
        else:
            self.specified_first_derivative_at_start = False

        if self.composition[-1][2] == 'c' and len(self.composition) != 1:
            # if the last motif is finite (and it's not the only motif) we have an additional degree of freedom
            # for the first derivative at the end
            # with infinite composition, this is not needed as the last finite transition point is always an extremum or transition point
            self.first_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
            self.specified_first_derivative_at_end = True
            self.specified_second_derivative_at_end = False
        elif self.infinite_composition and len(self.composition) == 1:
            # if there is only one infinite motif, we need to specify the first derivative at the end
            # and possibly the second as well
            self.first_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
            self.specified_first_derivative_at_end = True
            second_derivative_vanishes = self.infinite_motif_classes[self.infinite_motif].second_derivative_vanishes()
            if not second_derivative_vanishes:
                self.second_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
                self.specified_second_derivative_at_end = True
            else:
                self.specified_second_derivative_at_end = False
        else:
            self.specified_first_derivative_at_end = False
            self.specified_second_derivative_at_end = False

        for key, value in self.scalers.items():
            if key in ['vertical', 'infinite_motif_start']:
                self.scalers[key] = torch.ones_like(value, requires_grad=False).to(self.device)
        
        self.scalers = torch.nn.ParameterDict(self.scalers)



    def number_of_properties_for_infinite_motif(self):
        if not self.infinite_composition:
            return 0
        else:
            return self.infinite_motif_classes[self.infinite_motif].num_network_properties()
            
    def extract_first_derivative_at_start(self,X,B):
        if self.specified_first_derivative_at_start:

            finite_coordinates = self.extract_coordinates_finite_composition(X, B) # shape (batch_size, n_all_coordinates, 2)

            motif_index = 0
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            calculated_first_derivative_ratio = torch.sigmoid(B @ self.first_derivative_at_start).flatten()
            slope_min, slope_max = self.get_first_derivative_range(0, coordinate_1, coordinate_2, "left")
            return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
                
        else:
            return None
        
    def extract_first_derivative_at_end(self,X,B):

        if self.composition[0][2] == 'c':
            # there is at least one finite motif

            if self.type_of_transition_point(len(self.composition_finite_part)) == 'max' or self.type_of_transition_point(len(self.composition_finite_part)) == 'min':
                return torch.zeros(X.shape[0]).to(self.device)

            all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
            last_transition_point = finite_coordinates[:,-1,:]

            last_first_derivative = 3*all_coefficients[:,-1,0] * last_transition_point[:,0]**2 + 2 * all_coefficients[:,-1,1] * last_transition_point[:,0] + all_coefficients[:,-1,2]
            return last_first_derivative
        elif self.specified_first_derivative_at_end:
            if self.infinite_composition and len(self.composition) == 1:
                sign = 1 if self.composition[-1][0] == '+' else -1
                return sign * torch.nn.functional.softplus(B @ self.first_derivative_at_end)
            else:
                pass
                # TODO: implement this for finite compositions

        
    def extract_second_derivative_at_end(self,X,B):
        if self.composition[0][2] == 'c':
            # there is at least one finite motif

            if self.type_of_transition_point(len(self.composition_finite_part)) == 'inflection':
                return torch.zeros(X.shape[0]).to(self.device)

            all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
            last_transition_point = finite_coordinates[:,-1,:]

            last_second_derivative = 6*all_coefficients[:,-1,0] * last_transition_point[:,0] + 2 * all_coefficients[:,-1,1]
            return last_second_derivative
        elif self.specified_second_derivative_at_end:
            sign = 1 if self.composition[-1][1] == '+' else -1
            return sign * torch.nn.functional.softplus(B @ self.second_derivative_at_end)
        else:
            return torch.zeros(X.shape[0]).to(self.device)

    
    def extract_properties_infinite_motif(self, X, B, t_range):
        if not self.infinite_composition:
            return torch.zeros(X.shape[0], self.number_of_properties_for_infinite_motif()).to(self.device)

        properties = torch.nn.functional.softplus(B @ self.infinite_motif_properties_weights) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties']) 

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
        last_transition_point = finite_coordinates[:,-1,:]

        infinite_motif_index = len(self.composition) - 1

        if infinite_motif_index > 0:
            if self.type_of_transition_point(len(self.composition_finite_part)) == 'inflection':
                last_second_derivative = torch.zeros(X.shape[0]).to(self.device)
            else:
                last_second_derivative = 6*all_coefficients[:,[infinite_motif_index-1],0] * last_transition_point[:,[0]] + 2 * all_coefficients[:,[infinite_motif_index-1],1]
            if self.type_of_transition_point(len(self.composition_finite_part)) == 'max' or self.type_of_transition_point(len(self.composition_finite_part)) == 'min':
                last_first_derivative = torch.zeros(X.shape[0]).to(self.device)
            else:
                last_first_derivative = 3*all_coefficients[:,[infinite_motif_index-1],0] * last_transition_point[:,[0]]**2 + 2 * all_coefficients[:,[infinite_motif_index-1],1] * last_transition_point[:,[0]] + all_coefficients[:,[infinite_motif_index-1],2]
        else:
            # there is only one motif
            sign = 1 if self.composition[-1][0] == '+' else -1
            last_first_derivative = sign * torch.nn.functional.softplus(B @ self.first_derivative_at_end)
            if self.specified_second_derivative_at_end:
                sign = 1 if self.composition[-1][1] == '+' else -1
                last_second_derivative = sign * torch.nn.functional.softplus(B @ self.second_derivative_at_end)
            else:
                last_second_derivative = torch.zeros(X.shape[0]).to(self.device)
            

        x0 = last_transition_point[:,[0]]
        y0 = last_transition_point[:,[1]]

        if self.fixed_infinite_properties is not None:
            for index, property_function in self.fixed_infinite_properties.items():
                properties[:,index] = property_function(X, x0, y0, last_first_derivative, last_second_derivative)

        motif_class = self.infinite_motif_classes[self.infinite_motif]
        result = motif_class.extract_properties_from_network(properties, x0, y0, last_first_derivative, last_second_derivative)
        return result
    
    def extract_coordinates_finite_composition(self, X, B):

        if self.infinite_composition:
            if len(self.composition) > 1:
                t_last_finite_transition_point = (B @ self.infinite_motif_start_weights * self.scalers['infinite_motif_start']).flatten()
                empirical_range = self.t_range[1] - self.t_range[0]
                t_last_finite_transition_point = self.t_range[0] + MIN_TRANSITION_POINT_SEP + torch.sigmoid(t_last_finite_transition_point) * (1-MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT) * empirical_range
            else:
                t_last_finite_transition_point = self.t_range[0] * torch.ones(X.shape[0]).to(self.device)
        else:
            t_last_finite_transition_point = self.t_range[1] * torch.ones(X.shape[0]).to(self.device)

        calculated_values_horizontal = (B @ self.horizontal_weights)
        calculated_values_vertical = (B @ self.vertical_weights)


        all_coordinate_values = torch.zeros(X.shape[0], len(self.composition_finite_part)+1,2).to(self.device)

        # pass horizontal values through a softmax function, and scale them by the range of the time points
        scale = t_last_finite_transition_point - self.t_range[0]

        calculated_values_horizontal = F.softmax(calculated_values_horizontal, dim=1) * (scale.reshape(-1,1).to(self.device))

        # make sure vertical values are positive
        calculated_values_vertical = torch.nn.functional.softplus(calculated_values_vertical * self.scalers['vertical'])
        
        # j = 0
        all_coordinate_values[:,0,0] = self.t_range[0]
        all_coordinate_values[:,0,1] = X[:,0]

        for j in range(1, len(self.composition_finite_part)+1):
            sign = 1 if self.composition_finite_part[j-1][0] == '+' else -1
            all_coordinate_values[:,j,0] = calculated_values_horizontal[:,j-1] + all_coordinate_values[:,j-1,0]
            all_coordinate_values[:,j,1] = sign*calculated_values_vertical[:,j-1] + all_coordinate_values[:,j-1,1]

        # Force the last point to be the end of the time range
        all_coordinate_values[:,-1,0] = t_last_finite_transition_point

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
        slope = (point2[:,1] - point1[:,1])/(point2[:,0] - point1[:,0])
        motif = self.composition[motif_index]

        if which_point == 'left':
            if motif == '++c':
                if self.type_of_transition_point(motif_index+1) == 'end':
                    return torch.zeros_like(slope).to(self.device), slope
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return torch.zeros_like(slope).to(self.device), slope
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
                    return torch.zeros_like(slope).to(self.device), slope
                elif self.type_of_transition_point(motif_index+1) == 'inflection':
                    return torch.zeros_like(slope).to(self.device), slope
        
        elif which_point == 'right':
            if motif == '++c':
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return slope, 3*slope
                elif self.type_of_transition_point(motif_index) == 'min':
                    return 1.5 * slope, 3 * slope
            elif motif == "+-c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return torch.zeros_like(slope).to(self.device), slope
            elif motif == "-+c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return torch.zeros_like(slope).to(self.device), slope
            elif motif == "--c":
                if self.type_of_transition_point(motif_index) == 'inflection':
                    return slope, 3 * slope
                elif self.type_of_transition_point(motif_index) == 'max':
                    return 1.5 * slope, 3 * slope
                
        # in general left of -a and right of +a are the same and vice versa.
                
    def get_coefficients_and_coordinates_finite_composition(self, X, B):

        finite_coordinates = self.extract_coordinates_finite_composition(X, B) # shape (batch_size, n_all_coordinates, 2)
        
        b = torch.zeros(X.shape[0], 3).to(self.device)
        coefficients_list = []

        for motif_index, motif in enumerate(self.composition_finite_part):
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            A_row_0 = torch.stack([coordinate_1[:,0]**3, coordinate_1[:,0]**2, coordinate_1[:,0], torch.ones(X.shape[0]).to(self.device)], dim=1)
            A_row_1 = torch.stack([coordinate_2[:,0]**3, coordinate_2[:,0]**2, coordinate_2[:,0], torch.ones(X.shape[0]).to(self.device)], dim=1)
            b_0 = coordinate_1[:,1]
            b_1 = coordinate_2[:,1]

            type_1 = self.type_of_transition_point(motif_index)
            type_2 = self.type_of_transition_point(motif_index+1)
            if type_1 == 'max' or type_1 == 'min':
                A_row_2 = torch.cat([3*coordinate_1[:,[0]]**2, 2*coordinate_1[:,[0]], torch.ones(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                b_2 = torch.zeros(X.shape[0]).to(self.device)
            elif type_1 == 'inflection':
                A_row_2 = torch.cat([6*coordinate_1[:,[0]], 2*torch.ones(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                b_2 = torch.zeros(X.shape[0]).to(self.device)
            elif type_1 == 'start' and self.specified_first_derivative_at_start:
                A_row_2 = torch.cat([3*coordinate_1[:,[0]]**2, 2*coordinate_1[:,[0]], torch.ones(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                calculated_first_derivative_ratio = torch.sigmoid(B @ self.first_derivative_at_start).flatten()
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "left")
                b_2 = slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
                if type_2 == 'end':
                    # in that case we reduce the cubic to a quadratic
                    A_row_3 = torch.cat([torch.ones(X.shape[0],1), torch.zeros(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                    b_3 = torch.zeros(X.shape[0]).to(self.device)
            if type_2 == 'max' or type_2 == 'min':
                A_row_3 = torch.cat([3*coordinate_2[:,[0]]**2, 2*coordinate_2[:,[0]], torch.ones(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                b_3 = torch.zeros(X.shape[0]).to(self.device)
            elif type_2 == 'inflection':
                A_row_3 = torch.cat([6*coordinate_2[:,[0]], 2*torch.ones(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                b_3 = torch.zeros(X.shape[0]).to(self.device)
            elif (type_2 == 'end' and self.specified_first_derivative_at_end) and type_1 != 'start':
                A_row_3 = torch.cat([3*coordinate_2[:,[0]]**2, 2*coordinate_2[:,[0]], torch.ones(X.shape[0], 1).to(self.device), torch.zeros(X.shape[0], 1).to(self.device)], dim=1)
                calculated_first_derivative_ratio = torch.sigmoid(B @ self.first_derivative_at_end).flatten()
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "right")
                b_3 = slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)

            A = torch.stack([A_row_0, A_row_1, A_row_2, A_row_3], dim=1)
            b = torch.stack([b_0, b_1, b_2, b_3], dim=1)

            if torch.any(torch.abs(torch.linalg.det(A)) < 1e-9):
                # print("exception")
                # just connect with a line
                slope = (coordinate_2[:,1] - coordinate_1[:,1])/(coordinate_2[:,0]-coordinate_1[:,0])
                b = coordinate_1[:,1] - slope * coordinate_1[:,0]
                coefficients = torch.stack([torch.zeros_like(b).to(self.device),torch.zeros_like(b).to(self.device),slope,b],dim=1)
            else:
                coefficients = torch.linalg.solve(A, b)
            
            coefficients_list.append(coefficients)
        if len(coefficients_list) == 0:
            all_coefficients = torch.zeros(X.shape[0], 0, 4).to(self.device)
        else:
            all_coefficients = torch.stack(coefficients_list, dim=1) # shape (batch_size, n_motifs, 4)
        return all_coefficients, finite_coordinates

    def evaluate_piece(self,finite_motif_coefficients, infinite_motif_properties, motif_index, T, last_transition_point=None,last_1st_derivative=None,last_2nd_derivative=None):
        if self.composition[motif_index][2] == 'c':
            # finite motif
            a = finite_motif_coefficients[:,[motif_index],0].repeat(1,T.shape[1])
            b = finite_motif_coefficients[:,[motif_index],1].repeat(1,T.shape[1])
            c = finite_motif_coefficients[:,[motif_index],2].repeat(1,T.shape[1])
            d = finite_motif_coefficients[:,[motif_index],3].repeat(1,T.shape[1])
            return a*T**3 + b*T**2 + c*T + d
        else:
            x0 = last_transition_point[:,[0]]
            y0 = last_transition_point[:,[1]]
            y1 = last_1st_derivative
            y2 = last_2nd_derivative

            # infinite motif
            motif_class = self.infinite_motif_classes[self.infinite_motif]
            T_to_use = torch.where(T < x0, x0, T) # make sure we don't evaluate the infinite motif before the last transition point - this might cause errors
            return motif_class.evaluate_from_network(T_to_use,infinite_motif_properties, x0, y0, y1, y2)
            
    def forward(self, X, B, T):
        """
        Forward pass of the model

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)
        T: input tensor of shape (batch_size, n_time_points)
        """

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
        knots = finite_coordinates[:,:,0]
        last_transition_point = finite_coordinates[:,-1,:]
        
        if self.infinite_composition:
            # add infinite knots
            knots = torch.cat([knots, torch.from_numpy(np.array([np.inf])).reshape(1,1).repeat(X.shape[0],1).to(self.device)],dim=1)
            properties = torch.nn.functional.softplus(B @ self.infinite_motif_properties_weights) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties'])
        else:
            properties = None

        Y_pred = torch.zeros(X.shape[0], T.shape[1]).to(self.device)

        for i in range(len(self.composition)):
            if self.composition[i][2] != 'c' and i > 0:
                if self.type_of_transition_point(i) == 'min' or self.type_of_transition_point(i) == 'max':
                    last_first_derivative = torch.zeros_like(X).to(self.device)
                else:
                    last_first_derivative = 3*all_coefficients[:,[i-1],0] * last_transition_point[:,[0]]**2 + 2 * all_coefficients[:,[i-1],1] * last_transition_point[:,[0]] + all_coefficients[:,[i-1],2]
                
                if self.type_of_transition_point(i) == 'inflection':
                    last_second_derivative = torch.zeros_like(X).to(self.device)
                else:
                    last_second_derivative = 6*all_coefficients[:,[i-1],0] * last_transition_point[:,[0]] + 2 * all_coefficients[:,[i-1],1]
            elif self.composition[i][2] != 'c' and i == 0:
                sign = 1 if self.composition[-1][0] == '+' else -1
                last_first_derivative = sign * torch.nn.functional.softplus(B @ self.first_derivative_at_end)
                if self.specified_second_derivative_at_end:
                    sign = 1 if self.composition[-1][1] == '+' else -1
                    last_second_derivative = sign * torch.nn.functional.softplus(B @ self.second_derivative_at_end)
                else:
                    last_second_derivative = torch.zeros_like(X).to(self.device)
            else:
                last_first_derivative = None
                last_second_derivative = None

            if self.fixed_infinite_properties is not None:
                x0 = last_transition_point[:,[0]]
                y0 = last_transition_point[:,[1]]
                y1 = last_first_derivative
                y2 = last_second_derivative
                for index, property_function in self.fixed_infinite_properties.items():
                    properties[:,index] = property_function(X, x0, y0, y1, y2)
            
     
            evaluated_piece = self.evaluate_piece(all_coefficients,properties,i,T,last_transition_point,last_first_derivative,last_second_derivative)

            Y_pred += torch.where((knots[:,[i]].repeat(1,T.shape[1]) <= T) & (T < knots[:,[i+1]].repeat(1,T.shape[1])),evaluated_piece,0)
        
        if not self.infinite_composition:
            # Due the sharp inequalities earlier, we need to add the last piece separately
            a = all_coefficients[:,[-1],0].repeat(1,T.shape[1])
            b = all_coefficients[:,[-1],1].repeat(1,T.shape[1])
            c = all_coefficients[:,[-1],2].repeat(1,T.shape[1])
            d = all_coefficients[:,[-1],3].repeat(1,T.shape[1])
            Y_pred += torch.where(T == knots[:,[-1]].repeat(1,T.shape[1]),a*T**3 + b*T**2 + c*T + d,0)
            # possibly add values beyond last t based on some condition, you can use first or second derivite information
        
        return Y_pred
    
    def _forward(self, all_coefficients, finite_coordinates, X, B, T):
        """
        Forward pass of the model

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)
        T: input tensor of shape (batch_size, n_time_points)
        """

        knots = finite_coordinates[:,:,0]
        last_transition_point = finite_coordinates[:,-1,:]
        
        if self.infinite_composition:
            # add infinite knots
            knots = torch.cat([knots, torch.from_numpy(np.array([np.inf])).reshape(1,1).repeat(X.shape[0],1).to(self.device)],dim=1)
            properties = torch.nn.functional.softplus(B @ self.infinite_motif_properties_weights) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties'])
        else:
            properties = None

        Y_pred = torch.zeros(X.shape[0], T.shape[1]).to(self.device)

        for i in range(len(self.composition)):
            if self.composition[i][2] != 'c' and i > 0:
                if self.type_of_transition_point(i) == 'min' or self.type_of_transition_point(i) == 'max':
                    last_first_derivative = torch.zeros_like(X).to(self.device)
                else:
                    last_first_derivative = 3*all_coefficients[:,[i-1],0] * last_transition_point[:,[0]]**2 + 2 * all_coefficients[:,[i-1],1] * last_transition_point[:,[0]] + all_coefficients[:,[i-1],2]
                
                if self.type_of_transition_point(i) == 'inflection':
                    last_second_derivative = torch.zeros_like(X).to(self.device)
                else:
                    last_second_derivative = 6*all_coefficients[:,[i-1],0] * last_transition_point[:,[0]] + 2 * all_coefficients[:,[i-1],1]
                
            elif self.composition[i][2] != 'c' and i == 0:
                sign = 1 if self.composition[-1][0] == '+' else -1
                last_first_derivative = sign * torch.nn.functional.softplus(B @ self.first_derivative_at_end)
                if self.specified_second_derivative_at_end:
                    sign = 1 if self.composition[-1][1] == '+' else -1
                    last_second_derivative = sign * torch.nn.functional.softplus(B @ self.second_derivative_at_end)
                else:
                    last_second_derivative = torch.zeros_like(X).to(self.device)
            else:
                last_first_derivative = None
                last_second_derivative = None
            
            if self.fixed_infinite_properties is not None:
                x0 = last_transition_point[:,[0]]
                y0 = last_transition_point[:,[1]]
                y1 = last_first_derivative
                y2 = last_second_derivative
                for index, property_function in self.fixed_infinite_properties.items():
                    properties[:,index] = property_function(X, x0, y0, y1, y2)
     
            evaluated_piece = self.evaluate_piece(all_coefficients,properties,i,T,last_transition_point,last_first_derivative,last_second_derivative)

            Y_pred += torch.where((knots[:,[i]].repeat(1,T.shape[1]) <= T) & (T < knots[:,[i+1]].repeat(1,T.shape[1])),evaluated_piece,0)
         
        if not self.infinite_composition:
            # Due the sharp inequalities earlier, we need to add the last piece separately
            a = all_coefficients[:,[-1],0].repeat(1,T.shape[1])
            b = all_coefficients[:,[-1],1].repeat(1,T.shape[1])
            c = all_coefficients[:,[-1],2].repeat(1,T.shape[1])
            d = all_coefficients[:,[-1],3].repeat(1,T.shape[1])
            Y_pred += torch.where(T == knots[:,[-1]].repeat(1,T.shape[1]),a*T**3 + b*T**2 + c*T + d,0)
            # possibly add values beyond last t based on some condition, you can use first or second derivite information
        
        return Y_pred
    
    def loss_discontinuity_of_derivatives(self, coefficients, finite_coordinates):


        def derivative_difference_loss():

            global_first_derivative_discontinuity = 0
            global_second_derivative_discontinuity = 0
            
            for i in range(len(self.composition_finite_part)-1):
                if self.type_of_transition_point(i+1) == 'min' or self.type_of_transition_point(i+1) == 'max':
                    pass
                else:
                    first_derivatives_left = 3*coefficients[:,i,0] * finite_coordinates[:,i+1,0]**2 + 2 * coefficients[:,i,1] * finite_coordinates[:,i+1,0] + coefficients[:,i,2]
                    first_derivatives_right = 3*coefficients[:,i+1,0] * finite_coordinates[:,i+1,0]**2 + 2 * coefficients[:,i+1,1] * finite_coordinates[:,i+1,0] + coefficients[:,i+1,2]

                    first_derivative_discontinuity = first_derivatives_left - first_derivatives_right

                    first_norm = torch.max(torch.abs(first_derivatives_left), torch.abs(first_derivatives_right))
                    mask = first_norm > 1e-3

                    first_derivative_discontinuity[mask] =  first_derivative_discontinuity[mask]/first_norm[mask]
                    first_derivative_discontinuity[~mask] = first_derivative_discontinuity[~mask]

                    global_first_derivative_discontinuity += torch.mean(first_derivative_discontinuity ** 2)
                if self.type_of_transition_point(i+1) == 'inflection':
                    pass
                else:
                    second_derivatives_left = 6*coefficients[:,i,0] * finite_coordinates[:,i+1,0] + 2 * coefficients[:,i,1]
                    second_derivatives_right = 6*coefficients[:,i+1,0] * finite_coordinates[:,i+1,0] + 2 * coefficients[:,i+1,1]

                    second_derivative_discontinuity = second_derivatives_left - second_derivatives_right

                    second_norm = torch.max(torch.abs(second_derivatives_left), torch.abs(second_derivatives_right))
                    mask = second_norm > 1e-3

                    second_derivative_discontinuity[mask] =  second_derivative_discontinuity[mask]/second_norm[mask]
                    second_derivative_discontinuity[~mask] = second_derivative_discontinuity[~mask]*1e3

                    global_second_derivative_discontinuity += torch.mean(second_derivative_discontinuity ** 2)

            # calculate the loss
            return global_first_derivative_discontinuity + global_second_derivative_discontinuity

        def last_derivative_loss():
             # calculate the first derivative at the transition points
            first_derivative_last = 3*coefficients[:,-1,0] * finite_coordinates[:,-1,0]**2 + 2 * coefficients[:,-1,1] * finite_coordinates[:,-1,0] + coefficients[:,-1,2]
            return torch.mean(first_derivative_last ** 2)


        if finite_coordinates.shape[1] <= 1:
            return 0
        elif finite_coordinates.shape[1] <=2:
            return last_derivative_loss() * self.dis_loss_coeff_2
        else:
            return derivative_difference_loss() * self.dis_loss_coeff_1 + last_derivative_loss() * self.dis_loss_coeff_2

    
    def loss(self, X, B, T, Y, mask=False, with_derivative_loss=True):
        """
        Compute the loss function

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)
        T: input tensor of shape (batch_size, n_time_points)
        Y: input tensor of shape (batch_size, n_time_points)
        """

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
       
        # mask = False
        Y_pred = self._forward(all_coefficients, finite_coordinates, X, B, T) # shape (batch_size, n_time_points)
        # print(Y_pred)
   
        loss_per_sample = torch.sum((Y_pred - Y) ** 2, dim=1) / Y_pred.shape[1] # shape (batch_size,)
       
        normalize = False
        if normalize:
            range_per_sample = torch.max(Y, dim=1).values - torch.min(Y, dim=1).values
            normalized_loss_per_sample = loss_per_sample / range_per_sample
            loss = torch.mean(normalized_loss_per_sample)
        else:
            loss = torch.mean(loss_per_sample)

        if with_derivative_loss:
            derivative_loss = self.loss_discontinuity_of_derivatives(all_coefficients, finite_coordinates)
            return loss + derivative_loss
        else:
            return loss
    