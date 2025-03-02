import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline
import numpy as np
from semantic_odes.infinite_motifs import *
from semantic_odes import utils

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

        self.torch_device = utils.get_torch_device(config['device'])

        if config['t_range'][1] is None:
            self.t_range = torch.Tensor([config['t_range'][0],np.inf]).to(self.torch_device)
        else:
            self.t_range = torch.Tensor([config['t_range'][0],config['t_range'][1]]).to(self.torch_device)
        torch.manual_seed(self.seed)

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

        self.scalers = {}

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

        self.first_derivative_at_start_status = utils.get_first_derivative_at_start_status(self.composition)
        self.first_derivative_at_end_status = utils.get_first_derivative_at_end_status(self.composition)
        self.second_derivative_at_end_status = utils.get_second_derivative_at_end_status(self.composition)

        if self.first_derivative_at_start_status == 'weights':
            self.first_derivative_at_start = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
        
        if self.first_derivative_at_end_status == 'weights':
            self.first_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))
        
        if self.second_derivative_at_end_status == 'weights':
            self.second_derivative_at_end = torch.nn.Parameter(torch.randn(self.n_basis_functions, 1))

        for key, value in self.scalers.items():
            if key in ['vertical', 'infinite_motif_start']:
                self.scalers[key] = torch.ones_like(value, requires_grad=False).to(self.torch_device)
        
        self.scalers = torch.nn.ParameterDict(self.scalers)

    def number_of_properties_for_infinite_motif(self):
        """
        Returns the number of properties for the infinite motif

        Returns:
        num_properties: number of properties for the infinite motif
        """
        if not self.infinite_composition:
            return 0
        else:
            return self.infinite_motif_classes[self.infinite_motif].num_network_properties()
            
    def extract_first_derivative_at_start(self,X,B, finite_coordinates=None):
        """
        Extracts the first derivative at the start of the composition

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)
        finite_coordinates: input tensor of shape (batch_size, n_finite_coordinates, 2)

        Returns:
        first_derivative_at_start: tensor of shape (batch_size,)
        """
        if self.first_derivative_at_start_status == 'weights':
            if finite_coordinates is None:
                finite_coordinates = self.extract_coordinates_finite_composition(X, B) # shape (batch_size, n_all_coordinates, 2)

            motif_index = 0
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            calculated_first_derivative_ratio = torch.sigmoid(B @ self.first_derivative_at_start).flatten()
            slope_min, slope_max = self.get_first_derivative_range(0, coordinate_1, coordinate_2, "left")
            return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
        else:
            return None
        
    def extract_first_derivative_at_end(self,X,B, all_coefficients=None, finite_coordinates=None):
        """
        Extracts the first derivative at the end of the finite composition

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)
        all_coefficients: input tensor of shape (batch_size, n_cubics, 4)
        finite_coordinates: input tensor of shape (batch_size, n_finite_coordinates, 2)

        Returns:
        first_derivative_at_end: tensor of shape (batch_size,)
        """

        if self.first_derivative_at_end_status == 'zero':
            return torch.zeros(X.shape[0]).to(self.torch_device)
        elif self.first_derivative_at_end_status == 'cubic':
            if (all_coefficients is None) or (finite_coordinates is None):
                all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
            last_transition_point = finite_coordinates[:,-1,:]

            last_first_derivative = 3*all_coefficients[:,-1,0] * last_transition_point[:,0]**2 + 2 * all_coefficients[:,-1,1] * last_transition_point[:,0] + all_coefficients[:,-1,2]
            return last_first_derivative
        elif self.first_derivative_at_end_status == 'weights':
            if utils.is_unbounded_composition(self.composition):
                sign = 1 if self.composition[-1][0] == '+' else -1
                return sign * torch.nn.functional.softplus(B @ self.first_derivative_at_end)
            else:
                if finite_coordinates is None:
                    finite_coordinates = self.extract_coordinates_finite_composition(X, B)
                motif_index = len(self.composition) - 1
                coordinate_1 = finite_coordinates[:,motif_index,:]
                coordinate_2 = finite_coordinates[:,motif_index+1,:]
                calculated_first_derivative_ratio = torch.sigmoid(B @ self.first_derivative_at_end).flatten()
                slope_min, slope_max = self.get_first_derivative_range(motif_index, coordinate_1, coordinate_2, "right")
                return slope_min + calculated_first_derivative_ratio * (slope_max - slope_min)
        else:
            return None

        
    def extract_second_derivative_at_end(self,X,B, all_coefficients=None, finite_coordinates=None):
        """
        Extracts the second derivative at the end of the finite composition.
        Parameters:
        X (torch.Tensor): Input tensor of shape (batch_size, 1).
        B (torch.Tensor): Input tensor of shape (batch_size, n_basis_functions).
        all_coefficients (torch.Tensor, optional): input tensor of shape (batch_size, n_motifs, 4) Defaults to None.
        finite_coordinates (torch.Tensor, optional): input tensor of shape (batch_size, n_all_coordinates, 2). Defaults to None.
        Returns:
        torch.Tensor: tensor of shape (batch_size,). The output depends on the status of 
                      `self.second_derivative_at_end_status`:
                      - 'zero': Returns a tensor of zeros with the same batch size as X.
                      - 'cubic': Returns the second derivative computed using cubic coefficients.
                      - 'weights': Returns the second derivative computed using weighted softplus function.
                      - Otherwise: Returns None.
        """

        if self.second_derivative_at_end_status == 'zero':
            return torch.zeros(X.shape[0]).to(self.torch_device)
        elif self.second_derivative_at_end_status == 'cubic':
            if (all_coefficients is None) or (finite_coordinates is None):
                all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
            last_transition_point = finite_coordinates[:,-1,:]

            last_second_derivative = 6*all_coefficients[:,-1,0] * last_transition_point[:,0] + 2 * all_coefficients[:,-1,1]
            return last_second_derivative
        elif self.second_derivative_at_end_status == "weights":
            sign = 1 if self.composition[-1][1] == '+' else -1
            return sign * torch.nn.functional.softplus(B @ self.second_derivative_at_end)
        else:
            return None
    
    def extract_properties_infinite_motif(self, X, B, t_range):
        """
        Extracts properties for an infinite motif based on the input data and parameters.
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, 1).
            B (torch.Tensor): Input tensor of shape (batch_size, n_basis_functions).
            t_range (torch.Tensor): Tensor representing the range of time points.
        Returns:
            torch.Tensor: Tensor containing the properties of the infinite motif. Shape is (batch_size, num_properties).
        Notes:
            - If `self.infinite_composition` is False, returns a tensor of zeros with the appropriate shape.
            - If `self.fixed_infinite_properties` is not None, updates the properties tensor using the provided property functions.
        """

        if not self.infinite_composition:
            return torch.zeros(X.shape[0], self.number_of_properties_for_infinite_motif()).to(self.torch_device)

        properties = torch.nn.functional.softplus(B @ self.infinite_motif_properties_weights) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties']) 

        all_coefficients, finite_coordinates = self.get_coefficients_and_coordinates_finite_composition(X, B)
        last_transition_point = finite_coordinates[:,-1,:]

        last_first_derivative = self.extract_first_derivative_at_end(X, B, all_coefficients, finite_coordinates).reshape(-1,1)
        last_second_derivative = self.extract_second_derivative_at_end(X, B, all_coefficients, finite_coordinates).reshape(-1,1)

        x0 = last_transition_point[:,[0]]
        y0 = last_transition_point[:,[1]]

        if self.fixed_infinite_properties is not None:
            for index, property_function in self.fixed_infinite_properties.items():
                properties[:,index] = property_function(X, x0, y0, last_first_derivative, last_second_derivative)

        motif_class = self.infinite_motif_classes[self.infinite_motif]
        result = motif_class.extract_properties_from_network(properties, x0, y0, last_first_derivative, last_second_derivative)
        return result
    
    def extract_coordinates_finite_composition(self, X, B):
        """
        Extracts the coordinates of the finite composition based on X and B.

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)

        Returns:
        all_coordinate_values: tensor of shape (batch_size, n_all_coordinates, 2)
        """

        if self.infinite_composition:
            if len(self.composition) > 1:
                t_last_finite_transition_point = (B @ self.infinite_motif_start_weights * self.scalers['infinite_motif_start']).flatten()
                empirical_range = self.t_range[1] - self.t_range[0]
                t_last_finite_transition_point = self.t_range[0] + MIN_TRANSITION_POINT_SEP + torch.sigmoid(t_last_finite_transition_point) * (1-MIN_RELATIVE_DISTANCE_TO_LAST_FINITE_TRANSITION_POINT) * empirical_range
            else:
                t_last_finite_transition_point = self.t_range[0] * torch.ones(X.shape[0]).to(self.torch_device)
        else:
            t_last_finite_transition_point = self.t_range[1] * torch.ones(X.shape[0]).to(self.torch_device)

        calculated_values_horizontal = (B @ self.horizontal_weights)
        calculated_values_vertical = (B @ self.vertical_weights)


        all_coordinate_values = torch.zeros(X.shape[0], len(self.composition_finite_part)+1,2).to(self.torch_device)

        # pass horizontal values through a softmax function, and scale them by the range of the time points
        scale = t_last_finite_transition_point - self.t_range[0]

        calculated_values_horizontal = F.softmax(calculated_values_horizontal, dim=1) * (scale.reshape(-1,1).to(self.torch_device))

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
        return utils.type_of_transition_point(composition, ind)
      

    def get_first_derivative_range(self, motif_index, point1, point2, which_point):
        slope = (point2[:,1] - point1[:,1])/(point2[:,0] - point1[:,0])
        coefficients = utils.get_first_derivative_range_coefficients(self.composition, motif_index, which_point)
        return coefficients[0] * slope, coefficients[1] * slope
    
    def _create_row(self, coordinate, order):
        """
        Creates a row to determine the coefficients of the cubic

        Args:
        coordinate: input tensor of shape (batch_size, 2)
        order: order of the row (0, 1, 2)

        Returns:
        row: row tensor of shape (batch_size, 4)
        """
        batch_size = coordinate.shape[0]
        if order == 0:
            return torch.stack([coordinate[:,0]**3, coordinate[:,0]**2, coordinate[:,0], torch.ones(batch_size).to(self.torch_device)], dim=1)
        elif order == 1:
            return torch.cat([3*coordinate[:,[0]]**2, 2*coordinate[:,[0]], torch.ones(batch_size, 1).to(self.torch_device), torch.zeros(batch_size, 1).to(self.torch_device)], dim=1)
        elif order == 2:
            return torch.cat([6*coordinate[:,[0]], 2*torch.ones(batch_size, 1).to(self.torch_device), torch.zeros(batch_size, 1).to(self.torch_device), torch.zeros(batch_size, 1).to(self.torch_device)], dim=1)
                
                
    def get_coefficients_and_coordinates_finite_composition(self, X, B):
        """
        Get the coefficients of the cubics

        Args:
        X: input tensor of shape (batch_size, 1)
        B: input tensor of shape (batch_size, n_basis_functions)

        Returns:
        all_coefficients: tensor of shape (batch_size, n_cubics, 4)
        finite_coordinates: tensor of shape (batch_size, n_finite_coordinates, 2)
        """

        finite_coordinates = self.extract_coordinates_finite_composition(X, B) # shape (batch_size, n_all_coordinates, 2)
        
        b = torch.zeros(X.shape[0], 3).to(self.torch_device)
        coefficients_list = []

        for motif_index, motif in enumerate(self.composition_finite_part):
            coordinate_1 = finite_coordinates[:,motif_index,:]
            coordinate_2 = finite_coordinates[:,motif_index+1,:]

            A_row_0 = self._create_row(coordinate_1, 0)
            A_row_1 = self._create_row(coordinate_2, 0)
            b_0 = coordinate_1[:,1]
            b_1 = coordinate_2[:,1]

            type_1 = self.type_of_transition_point(motif_index)
            type_2 = self.type_of_transition_point(motif_index+1)
            if type_1 == 'max' or type_1 == 'min':
                A_row_2 = self._create_row(coordinate_1, 1)
                b_2 = torch.zeros(X.shape[0]).to(self.torch_device)
            elif type_1 == 'inflection':
                A_row_2 = self._create_row(coordinate_1, 2)
                b_2 = torch.zeros(X.shape[0]).to(self.torch_device)
            elif type_1 == 'start' and self.first_derivative_at_start_status == 'weights':
                A_row_2 = self._create_row(coordinate_1, 1)
                b_2 = self.extract_first_derivative_at_start(X,B,finite_coordinates)
                if type_2 == 'end':
                    # in that case we reduce the cubic to a quadratic
                    A_row_3 = torch.cat([torch.ones(X.shape[0],1), torch.zeros(X.shape[0], 1), torch.zeros(X.shape[0], 1), torch.zeros(X.shape[0], 1)], dim=1).to(self.torch_device)
                    b_3 = torch.zeros(X.shape[0]).to(self.torch_device)
            if type_2 == 'max' or type_2 == 'min':
                A_row_3 = self._create_row(coordinate_2, 1)
                b_3 = torch.zeros(X.shape[0]).to(self.torch_device)
            elif type_2 == 'inflection':
                A_row_3 = self._create_row(coordinate_2, 2)
                b_3 = torch.zeros(X.shape[0]).to(self.torch_device)
            elif (type_2 == 'end' and self.first_derivative_at_end_status == 'weights') and type_1 != 'start':
                A_row_3 = self._create_row(coordinate_2, 1)
                b_3 = self.extract_first_derivative_at_end(X,B,finite_coordinates=finite_coordinates)

            A = torch.stack([A_row_0, A_row_1, A_row_2, A_row_3], dim=1)
            b = torch.stack([b_0, b_1, b_2, b_3], dim=1)

            if torch.any(torch.abs(torch.linalg.det(A)) < 1e-9):
                # print("exception")
                # just connect with a line
                slope = (coordinate_2[:,1] - coordinate_1[:,1])/(coordinate_2[:,0]-coordinate_1[:,0])
                b = coordinate_1[:,1] - slope * coordinate_1[:,0]
                coefficients = torch.stack([torch.zeros_like(b).to(self.torch_device),torch.zeros_like(b).to(self.torch_device),slope,b],dim=1)
            else:
                coefficients = torch.linalg.solve(A, b)
            
            coefficients_list.append(coefficients)
        if len(coefficients_list) == 0:
            all_coefficients = torch.zeros(X.shape[0], 0, 4).to(self.torch_device)
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
      
        return self._forward(all_coefficients, finite_coordinates, X, B, T)
    
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
            knots = torch.cat([knots, torch.from_numpy(np.array([np.inf])).reshape(1,1).repeat(X.shape[0],1).to(self.torch_device)],dim=1)
            properties = torch.nn.functional.softplus(B @ self.infinite_motif_properties_weights) * torch.nn.functional.softplus(self.scalers['infinite_motif_properties'])
        else:
            properties = None

        Y_pred = torch.zeros(X.shape[0], T.shape[1]).to(self.torch_device)

        for i in range(len(self.composition)):

            if utils.is_unbounded_composition(self.composition):
                last_first_derivative = self.extract_first_derivative_at_end(X,B,all_coefficients,finite_coordinates).reshape(-1,1)
                last_second_derivative = self.extract_second_derivative_at_end(X, B, all_coefficients, finite_coordinates).reshape(-1,1)
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
    