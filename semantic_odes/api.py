from abc import ABC, abstractmethod
import copy
from datetime import datetime
import multiprocessing
import os
import optuna
from scipy.interpolate import BSpline
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from semantic_odes.lit_module import LitSketchODE
import pandas as pd
import logging
from semantic_odes.reconstruct_cubic import PredictiveModel, ApproximatePredictiveModel
from semantic_odes.model_numpy import calculate_loss, calculate_loss_3_samples
from semantic_odes.infinite_motifs import get_default_motif_class
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def process_sample(info):
        sample, compositions, t_range, seed = info
        x0, t, y = sample
        sample_scores = {'x0':x0}
        for composition in compositions:
            loss, model = calculate_loss(composition, t_range, x0, t, y, seed=seed, evaluate_on_all_data=True)
            sample_scores[tuple(composition)] = loss
        return sample_scores

def _assign_to_mask(mask,target,input):
    if np.any(mask):
        if np.ndim(input) == 0 or type(input) == tuple or type(input) == str:
            input = [input]*mask.sum()
        ids = np.arange(len(target))[mask]
        counter = 0
        for i in ids:
            target[i] = input[counter]
            counter += 1

def format_composition(composition):
    formatted_motifs = []
    for motif in composition:
        # Replace 'c' with 'b' and 'f', 'p' with 'u'
        motif_string = str(motif).replace('c','b').replace('f','u').replace('p','u')
        motif_string = "s_{"+str(motif_string)+"}"
        formatted_motifs.append(motif_string)
    return fr"$({', '.join(formatted_motifs)})$"
class BasisFunctions(ABC):

    def __init__(self, n_basis):
        self.n_basis = n_basis

    @abstractmethod
    def compute(self, x, x_range):
        pass

class BSplineBasisFunctions(BasisFunctions):

    def __init__(self,n_basis, k=3, include_bias=True, include_linear=False):
        super().__init__(n_basis)
        self.k = k
        self.include_bias = include_bias
        self.include_linear = include_linear
    
    def compute(self,X,x0_range):

        if self.include_bias:
            n_b_basis = self.n_basis - 1
        else:
            n_b_basis = self.n_basis

        if self.include_linear:
            n_b_basis = n_b_basis - 1

        shape_knots = np.r_[[x0_range[0]]*self.k,np.linspace(x0_range[0], x0_range[1], n_b_basis-self.k+1),[x0_range[1]]*self.k]
       
        def singleton_vector(n,k):
            vector = np.zeros(n)
            vector[k] = 1
            return vector
        
        bsplines = [BSpline(shape_knots,singleton_vector(n_b_basis,k_index),k=self.k,extrapolate=False) for k_index in range(n_b_basis)]

        # bspline_basis_per_sample = [BSpline(shape_knots,singleton_vector(n_b_basis,k_index),k=self.k,extrapolate=False)(X.flatten()) for k_index in range(n_b_basis)]
        bspline_basis_per_sample = [bspline(X.flatten()) for bspline in bsplines]
        # fill na values with 0
        final_list = []
        for i, values in enumerate(bspline_basis_per_sample):
            below = X.flatten() <= x0_range[0]
            above = X.flatten() >= x0_range[1]
            values[below] = bsplines[i](x0_range[0])
            values[above] = bsplines[i](x0_range[1])
            # print(X.flatten())
            # print(values)
            if np.any(np.isnan(values)):
                print(f'Nan values found for basis function {i}')
                print(f'X: {X.flatten()}')
                print(f'Values: {values}')
            final_list.append(values)

        if self.include_bias:
            # add the constant basis function
            final_list.append(np.ones_like(X.flatten()))
        
        if self.include_linear:
            # add the linear basis function
            values = X.flatten()
            below = X.flatten() <= x0_range[0]
            above = X.flatten() >= x0_range[1]
            values[below] = x0_range[0]
            values[above] = x0_range[1]
            final_list.append(values)


        B = np.stack(final_list, axis=1)

        return B

class PolynomialBasisFunctions(BasisFunctions):

    def _init_(self,n_basis):
        super()._init_(n_basis)

    def compute(self,X,x0_range):
        X = X.flatten()
        bases = []
        bases.append(np.ones_like(X))
        order = self.n_basis - 1
        for i in range(1,order+1):
            bases.append(X ** i)
        
        return np.stack(bases,axis=1)
    
class SemanticRepresentation:

    def __init__(self,t_range,composition,coordinates_finite_composition,derivative_start,derivative_end,properties_infinite_motif,second_derivative_end=None):
        self.t_range = t_range
        self.composition = composition
        self.coordinates_finite_composition = coordinates_finite_composition
        self.derivative_start = derivative_start
        self.derivative_end = derivative_end
        self.properties_infinite_motif = properties_infinite_motif
        self.second_derivative_end = second_derivative_end

    def __repr__(self):
        return f"""Composition: {self.composition}
Coordinates:
{self.coordinates_finite_composition}
Derivative at start: {self.derivative_start}
Derivative at end: {self.derivative_end}
Properties of infinite motif:
{self.properties_infinite_motif}
Second derivative at end: {self.second_derivative_end}"""
    
    def copy(self):
        return SemanticRepresentation(self.t_range,self.composition,self.coordinates_finite_composition,self.derivative_start,self.derivative_end,self.properties_infinite_motif,self.second_derivative_end)


def create_full_composition_library(max_length,is_infinite):

    motif_succession_rules = {
        '+-':['--','++'],
        '-+':['--','++'],
        '--':['-+'],
        '++':['+-']
    }

    motif_infinite_types = {
        '++':['f'],
        '+-':['p','h'],
        '-+':['f','h'],
        '--':['f']
    }

    all_compositions = []
    # dfs graph search algorithm
    def dfs(current_composition):

        if len(current_composition) > 0: # We do not add empty composition
            if is_infinite and current_composition[-1][2] != 'c':
                all_compositions.append(current_composition)
                return # The last motif is infinite, we cannot add more motifs
            elif not is_infinite:
                all_compositions.append(current_composition)
            # If the is_infinite but the last motif is finite, it's not a valid composition, so we do not add it to the list

        if len(current_composition) == max_length:
            return

        def expand(new_motif):
            if is_infinite:
                # We can make it a final motif by adding an infinite extension
                for infinite_extension in motif_infinite_types[new_motif]:
                    dfs(current_composition.copy() + [new_motif + infinite_extension])
                # We can also add a finite extension if there is still space
                if len(current_composition) < max_length-1:
                    dfs(current_composition.copy() + [new_motif + 'c'])
            else:
                dfs(current_composition.copy() + [new_motif + 'c'])

        if len(current_composition) == 0:
            for new_motif in ['+-','--','-+','++']:
                expand(new_motif)
        else:
            for new_motif in motif_succession_rules[current_composition[-1][0:2]]:
                expand(new_motif)
           
    dfs([])
    return all_compositions


class Cell:

    def __init__(self):
        self.next_composition = {}
        self.total_loss = {}
        self.samples_without_change = {}
        self.x0_without_change = {}


class CompositionMap:

    def __init__(self,composition_map_list):
        """
        Args:
        composition_map_list: list of tuples (x0_range, composition) where x0_range is a tuple (previous_x0, next_x0) and composition is a tuple of motifs
        """
        self.composition_map_list = composition_map_list

        self._validate()

    
    def _validate(self):
        """
        Checks if the ranges are valid
        """
        if self.composition_map_list[0][0][0] != -np.inf:
            raise ValueError(f'First range is not valid: {self.composition_map_list[0][0][0]} should be -np.inf')
        if self.composition_map_list[-1][0][1] != np.inf:
            raise ValueError(f'Last range is not valid: {self.composition_map_list[-1][0][1]} should be np.inf')

        for i in range(len(self.composition_map_list)-1):
            (_, next_x0), _ = self.composition_map_list[i]
            (next_previous_x0, _), _ = self.composition_map_list[i+1]
            if next_x0 != next_previous_x0:
                raise ValueError(f'Ranges are not continuous: {next_x0} != {next_previous_x0}')

    def predict(self, X0, with_composition_index=False, reduce=True):
        """
        Predict the composition for a given initial conditions
        
        Args:
        X0: numpy array of shape (batch_size, 1) or (batch_size,) or a scalar value with the initial conditions

        Returns:
        1D np.array (or a single object) with the compositions
        """

        self._validate()

        is_scalar = np.isscalar(X0)

        # Treat X0 as a 2D array
        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)
        
        compositions = np.empty(X0.shape[0],dtype=object)
        composition_indices = np.empty(X0.shape[0],dtype=int)

        for index, (x0_range, composition) in enumerate(self.composition_map_list):
            mask = (X0 > x0_range[0]) & (X0 <= x0_range[1])
            _assign_to_mask(mask.flatten(),compositions,composition)
            composition_indices[mask.flatten()] = index
        
        if with_composition_index:
            if is_scalar and reduce:
                return compositions[0], composition_indices[0]
            else:
                return compositions, composition_indices
        else:
            if is_scalar and reduce:
                return compositions[0]
            else:
                return compositions
        

    def __repr__(self):
        result = ''
        for (previous_x0, next_x0), composition in self.composition_map_list:
            if next_x0 == np.inf:
                result += f'({previous_x0},+inf):{composition}\n'
            else:
                result += f'({previous_x0},{next_x0}]:{composition}\n'
            
        return result

    def __len__(self):
        return len(self.composition_map_list)


class SinglePropertyMap:

    def __init__(self,x0_range,x0_finite_range,composition,transition_point_predictor,derivative_predictor,infinite_motif_predictor):
        self.x0_range = x0_range
        self.x0_finite_range = x0_finite_range
        self.composition = composition
        self.transition_point_predictor = transition_point_predictor
        self.derivative_predictor = derivative_predictor
        self.infinite_motif_predictor = infinite_motif_predictor
        self.n_transition_points = len(transition_point_predictor) // 2


    def _validate_X0(self,X0):
        if np.any(X0 < self.x0_range[0]) or np.any(X0 > self.x0_range[1]):
            raise ValueError(f'Initial conditions out of range: {X0} not in {self.x0_range}')

    def predict_transition_point(self,X0,transition_point_index,coordinate,reduce=True):
        self._validate_X0(X0)

        is_scalar = np.isscalar(X0)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)

        result = self.transition_point_predictor[(transition_point_index,coordinate)](X0).flatten()

        if is_scalar and reduce:
            return result[0]
        else:
            return result
    
    def predict_all_transition_points(self,X0,reduce=True):
        self._validate_X0(X0)

        is_scalar = np.isscalar(X0)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)

        results = np.empty((X0.shape[0],self.n_transition_points,2),dtype=float)
        for i in range(self.n_transition_points):
            results[:,i,0] = self.transition_point_predictor[(i,'t')](X0).flatten()
            results[:,i,1] = self.transition_point_predictor[(i,'x')](X0).flatten()

        if is_scalar and reduce:
            return results[0]
        else:
            return results
        
    def predict_derivative(self,X0,boundary,order,reduce=True):
        self._validate_X0(X0)

        is_scalar = np.isscalar(X0)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)

        if self.derivative_predictor is None:
            result = np.empty(X0.shape[0])
            result.fill(np.nan)
        else:
            result = self.derivative_predictor[(boundary,order)](X0).flatten()

        if is_scalar and reduce:
            return result[0]
        else:
            return result
        
    def predict_infinite_motif_property(self,X0,property_index,reduce=True):
        self._validate_X0(X0)

        is_scalar = np.isscalar(X0)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)
        
        if self.infinite_motif_predictor is None:
            result = np.empty(X0.shape[0])
            result.fill(np.nan)
        else:
            result = self.infinite_motif_predictor[property_index](X0).flatten()

        if is_scalar and reduce:
            return result[0]
        else:
            return result
        
    def predict_all_infinite_motif_properties(self,X0,reduce=True):
        self._validate_X0(X0)

        is_scalar = np.isscalar(X0)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)

        if self.infinite_motif_predictor is None:
            results = np.empty((X0.shape[0],1))
            results.fill(np.nan)
        else:
            results = np.empty((X0.shape[0],len(self.infinite_motif_predictor)),dtype=float)
            for i in range(len(self.infinite_motif_predictor)):
                results[:,i] = self.infinite_motif_predictor[i](X0).flatten()
        
        if is_scalar and reduce:
            return results[0]
        else:
            return results
        
    def visualize(self,layout):
        if layout == '1d':
            fig = plt.figure(figsize=(20,5))
            self._visualize_1d_layout(fig)
            plt.show()
        elif layout == '2d':
            self._visualize_2d_layout()
        else:
            print('Invalid layout')


    def _visualize_1d_layout(self, fig):
        n_cols = 4
        n_rows = 1
        axs = fig.subplots(n_rows,n_cols)

        axs[0].set_title(r'Transition points ($t$-coordinates)')
        axs[1].set_title(r'Transition points ($x$-coordinates)')
        axs[2].set_title('Derivatives')

        if len(self.infinite_motif_predictor) > 0:
            motif_string = "s_{"+str(self.composition[-1])+"}"
            axs[3].set_title(fr'Properties of unbounded motif ${motif_string}$')
        else:
            axs[3].set_title('Properties of unbounded motif')

        axs[0].set_xlabel(r'$x_0$')
        axs[1].set_xlabel(r'$x_0$')
        axs[2].set_xlabel(r'$x_0$')
        axs[3].set_xlabel(r'$x_0$')
        
        # for i in range(len(self.infinite_motif_predictor)):
        #     axs[i//n_cols_infinite_motif,2+i%n_cols_infinite_motif].set_title(f'Property {i}')

        x_space = np.linspace(self.x0_finite_range[0],self.x0_finite_range[1],100)

        for i in range(self.n_transition_points):
            axs[0].plot(x_space,self.predict_transition_point(x_space,i,'t'),label=fr"$t_{i}$")
            axs[1].plot(x_space,self.predict_transition_point(x_space,i,'x'),label=fr"$x(t_{i})$")
        axs[2].plot(x_space,self.predict_derivative(x_space,'start',1),label=r'$\dot{x}(t_{\text{start}})$')
        axs[2].plot(x_space,self.predict_derivative(x_space,'end',1),label=r'$\dot{x}(t_{\text{end}})$')
        axs[2].plot(x_space,self.predict_derivative(x_space,'end',2),label=r'$\ddot{x}(t_{\text{end}})$')

        if len(self.infinite_motif_predictor) > 0:
            # There is infinite motif
            infinite_motif = self.composition[-1]
            is_single = (len(self.composition) == 1)
            motif_class = get_default_motif_class(infinite_motif)
            property_names = motif_class.get_property_names()

        for i in range(len(self.infinite_motif_predictor)):
            # axs[i//n_cols_infinite_motif,2+i//n_rows_infinite_motif].plot(x_space,self.predict_infinite_motif_property(x_space,i))
            axs[3].plot(x_space,self.predict_infinite_motif_property(x_space,i),label=property_names[i])
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()

        # Global title for the whole figure
        fig.suptitle(f'Property map for composition {format_composition(self.composition)}',fontsize=16)

        fig.subplots_adjust(top=0.8)

        # fig.title(f'Property map for composition {self.format_composition(self.composition)}')
        

        
    def _visualize_2d_layout(self):

        if len(self.infinite_motif_predictor) == 0:
            n_cols_infinite_motif = 0
            n_rows_infinite_motif = 0
        elif len(self.infinite_motif_predictor) <= 2:
            n_cols_infinite_motif = 1
            n_rows_infinite_motif = 2
        else:
            n_cols_infinite_motif = 2
            n_rows_infinite_motif = (len(self.infinite_motif_predictor)+1) // 2

        # n_rows  = max(2,n_rows_infinite_motif)
        # n_cols = 2 + n_cols_infinite_motif  
        n_rows = 2
        n_cols = 2
        fig, axs = plt.subplots(n_rows,n_cols,figsize=(n_cols*5,n_rows*4))

        axs[0,0].set_title(r'Transition points ($t$-coordinates)')
        axs[1,0].set_title(r'Transition points ($x$-coordinates)')
        axs[0,1].set_title('Derivatives')

        if len(self.infinite_motif_predictor) > 0:
            motif_string = "s_{"+str(self.composition[-1])+"}"
            axs[1,1].set_title(fr'Properties of unbounded motif ${motif_string}$')
        else:
            axs[1,1].set_title('Properties of unbounded motif')

        axs[0,0].set_xlabel(r'$x_0$')
        axs[0,1].set_xlabel(r'$x_0$')
        axs[1,0].set_xlabel(r'$x_0$')
        axs[1,1].set_xlabel(r'$x_0$')
        
        # for i in range(len(self.infinite_motif_predictor)):
        #     axs[i//n_cols_infinite_motif,2+i%n_cols_infinite_motif].set_title(f'Property {i}')

        x_space = np.linspace(self.x0_finite_range[0],self.x0_finite_range[1],100)

        for i in range(self.n_transition_points):
            axs[0,0].plot(x_space,self.predict_transition_point(x_space,i,'t'),label=fr"$t_{i}$")
            axs[1,0].plot(x_space,self.predict_transition_point(x_space,i,'x'),label=fr"$x(t_{i})$")
        axs[0,1].plot(x_space,self.predict_derivative(x_space,'start',1),label=r'$\dot{x}(t_{\text{start}})$')
        axs[0,1].plot(x_space,self.predict_derivative(x_space,'end',1),label=r'$\dot{x}(t_{\text{end}})$')
        axs[0,1].plot(x_space,self.predict_derivative(x_space,'end',2),label=r'$\ddot{x}(t_{\text{end}})$')

        if len(self.infinite_motif_predictor) > 0:
            # There is infinite motif
            infinite_motif = self.composition[-1]
            is_single = (len(self.composition) == 1)
            motif_class = get_default_motif_class(infinite_motif)
            property_names = motif_class.get_property_names()

        for i in range(len(self.infinite_motif_predictor)):
            # axs[i//n_cols_infinite_motif,2+i//n_rows_infinite_motif].plot(x_space,self.predict_infinite_motif_property(x_space,i))
            axs[1,1].plot(x_space,self.predict_infinite_motif_property(x_space,i),label=property_names[i])
        axs[0,0].legend()
        axs[1,0].legend()
        axs[0,1].legend()
        axs[1,1].legend()
        plt.show()
        

        

class SemanticPredictor:

    def __init__(self,composition_map,property_maps,t_range):

        self.composition_map = composition_map
        self.property_maps = property_maps
        self.t_range = t_range

    def _valdate(self):
        if len(self.composition_map) != len(self.property_maps):
            raise ValueError('Composition map and property maps have different lengths')


    def predict(self,X0, reduce=True):

        is_scalar = np.isscalar(X0)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)

        compositions, indices = self.composition_map.predict(X0,with_composition_index=True, reduce=False)

        results = np.empty(X0.shape[0],dtype=object)

        for i in range(len(self.composition_map)):
            mask = (indices == i)
            composition = self.composition_map.composition_map_list[i][1]
            X0_filtered = X0[mask]
            property_map = self.property_maps[i]

            transition_points = property_map.predict_all_transition_points(X0_filtered)
            derivative_start = property_map.predict_derivative(X0_filtered,'start',1)
            derivative_end = property_map.predict_derivative(X0_filtered,'end',1)
            second_derivative_end = property_map.predict_derivative(X0_filtered,'end',2)
            properties_infinite_motif = property_map.predict_all_infinite_motif_properties(X0_filtered)

            semantic_representations = []
            for j in range(X0_filtered.shape[0]):
                semantic_representations.append(SemanticRepresentation(self.t_range,composition,transition_points[j],derivative_start[j],derivative_end[j],properties_infinite_motif[j],second_derivative_end[j]))

            _assign_to_mask(mask,results,semantic_representations)

        if is_scalar and reduce:
            return results[0]
        else:
            return results
        
    def _visualize_composition(self,ax,composition_index,finite_t_range, if_single=False):
        x0_range, composition = self.composition_map.composition_map_list[composition_index]
        x0_finite_range = self.property_maps[composition_index].x0_finite_range

        x0 = (x0_finite_range[0] + x0_finite_range[1]) / 2

        semantic_representation = self.predict(x0)
        model = ApproximatePredictiveModel(semantic_representation)
        t = np.linspace(finite_t_range[0],finite_t_range[1],1000)
        pred = model.forward(t)
        ax.plot(t,pred)
        # ax.set_title(f'{composition} between {x0_range[0]} and {x0_range[1]}')
        ax.set_xlabel('t')
        ax.set_ylabel('x')

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

        if if_single:
            x_plotted_range = pred.max() - pred.min()
            t_plotted_range = t.max() - t.min()

            ratio = x_plotted_range / t_plotted_range * 0.5

            ax.set_aspect(ratio)    

        transition_points = self.property_maps[composition_index].predict_all_transition_points(x0)
        ax.scatter(transition_points[:,0],transition_points[:,1],c='red',label='Transition points')

        # # Increase y_lim to make sure the annotation is visible
        # y_lim = ax.get_ylim()
        # ax.set_ylim(y_lim[0],y_lim[1]+np.abs(y_lim[1]-y_lim[0])*0.4)
        # # Annotate each transition point with a pointing arrow
        # for i, txt in enumerate(transition_points):
        #     ax.annotate(fr'$(t_{i},x(t_{i}))$', (txt[0],txt[1]), textcoords="offset points", xytext=(25,25), ha='center', arrowprops=dict(arrowstyle='->'))
            
        
    def _visualize_composition_map(self,fig,finite_t_range):

        axs = fig.subplots(1,len(self.composition_map))
        axs = np.atleast_1d(axs)

        if len(axs) == 1:
            if_single = True
        else:
            if_single = False

        for i, (x0_range, composition) in enumerate(self.composition_map.composition_map_list):
            axs[i].set_title(f'{composition} between {x0_range[0]} and {x0_range[1]}')
            self._visualize_composition(axs[i],i,finite_t_range, if_single)
            
        # Add a space at the top for the title
        fig.subplots_adjust(top=0.7)

    def _visualize_composition_map_vertical(self,figs,finite_t_range):

        for i, (x0_range, composition) in enumerate(self.composition_map.composition_map_list):
            ax = figs[i].subplots(1,1)
            if x0_range[0] == -np.inf:
                x0_print_1 = r'-\infty'
            else:
                x0_print_1 = f"{x0_range[0]:.2f}"
            if x0_range[1] == np.inf:
                x0_print_2 = r'+\infty'
            else:
                x0_print_2 = f"{x0_range[1]:.2f}"
            if x0_range[1] == np.inf:
                ax.set_title(rf'{format_composition(composition)} if $'+x0_print_1+rf'< x_0 < '+x0_print_2+'$', fontsize=14)
            else:
                ax.set_title(rf'{format_composition(composition)} if $'+x0_print_1+rf'< x_0 \leq '+x0_print_2+'$', fontsize=14)  
            # Add space between the title and the plot

            self._visualize_composition(ax,i,finite_t_range, False)
            # Add some space at the top and left
            figs[i].subplots_adjust(top=0.7, left=0.2)
            



    def visualize(self, finite_t_range, property_map_figsize=(20,5), composition_map_figsize=(20,2.5)):

        if property_map_figsize[0] != composition_map_figsize[0]:
            raise ValueError('Property map and composition map should have the same width')

        n_subfigures = 1 + len(self.property_maps)

        fig = plt.figure(figsize=(property_map_figsize[0],property_map_figsize[1]*(n_subfigures-1)+composition_map_figsize[1]))
        subfigs = fig.subfigures(n_subfigures,1, height_ratios=[composition_map_figsize[1]/property_map_figsize[1]] + [1]*len(self.property_maps), hspace=0.1)

        self._visualize_composition_map(subfigs[0],finite_t_range)
        subfigs[0].suptitle('Composition map', fontsize=16)

        for i, property_map in enumerate(self.property_maps):
            property_map._visualize_1d_layout(subfigs[i+1])

    def visualize_thin(self, finite_t_range, property_map_figsize=(20,5), composition_map_figsize=(5,5)):

        if property_map_figsize[1] != composition_map_figsize[1]:
            raise ValueError('Property map and composition map should have the same height')


        fig = plt.figure(figsize=(property_map_figsize[0]+composition_map_figsize[0],property_map_figsize[1]*(len(self.property_maps))))

        subfigs = fig.subfigures(len(self.property_maps),2, width_ratios=[composition_map_figsize[0]/property_map_figsize[0]] + [1], hspace=0.1, wspace=-0.25, squeeze=False)

        self._visualize_composition_map_vertical(subfigs[:,0],finite_t_range)
        # subfigs[0].suptitle('Composition map', fontsize=16)
        subfigs[0,0].suptitle('Composition map', fontsize=16)

        for i, property_map in enumerate(self.property_maps):
            property_map._visualize_1d_layout(subfigs[i,1])


class SemanticODE:

    def __init__(self,t_range,basis_functions,composition_library,max_branches,seed=0,opt_config={},verbose=False,x0_range=None):

        self.config = self._get_updated_opt_config(opt_config)
        print(self.config)
        
        self.t_range = t_range
        self.x0_range = x0_range
        self.basis_functions = basis_functions
        self.composition_library = composition_library
        self.max_branches = max_branches
        self.verbose = verbose

        self.config['t_range'] = t_range
        self.config['n_basis_functions'] = basis_functions.n_basis
        self.config['seed'] = seed

        self.composition_scores_df = None
        self.composition_map = None
        self.torch_models = None

    def _get_updated_opt_config(self,opt_config):
        config = self._get_default_opt_config()
        for key, value in opt_config.items():
            config[key] = value
        return config
    

    def _get_tensors(self,*args):
        """
        Convert numpy arrays to torch tensors

        Args:
        args: numpy arrays

        Returns:
        tuple of torch tensors
        """
        return tuple([torch.tensor(arg, dtype=torch.float32, device=self.config['device']) for arg in args])
    
    def _finite_x0_range(self,prev_x0,next_x0):
        """
        Get the range of the initial condition

        Args:
        X0: initial condition numpy array of shape (batch_size, 1)

        Returns:
        tuple of the range of the initial condition
        """
        if prev_x0 is None:
            prev_x0 = -np.inf
        if prev_x0 > -np.inf:
            x0_range_0 = prev_x0
        elif self.x0_range is not None:
            x0_range_0 = self.x0_range[0]

        if next_x0 is None:
            next_x0 = np.inf
        if next_x0 < np.inf:
            x0_range_1 = next_x0
        elif self.x0_range is not None:
            x0_range_1 = self.x0_range[1]

        x0_range = (x0_range_0, x0_range_1)
        return x0_range
    
    def _get_global_x0_range(self,X0):
        """
        Get the range of the initial condition

        Args:
        X0: initial condition numpy array of shape (batch_size, 1)

        Returns:
        tuple of the range of the initial condition
        """
        if self.x0_range is None:
            self.x0_range = (np.min(X0), np.max(X0))
        return self.x0_range
    
    def _get_train_val_indices(self,n_samples):
        """
        Get the indices for the training and validation sets

        Args:
        n_samples: number of samples

        Returns:
        tuple of numpy arrays with the indices
        """
        np_gen = np.random.default_rng(self.config['seed'])
        train_indices = np_gen.choice(n_samples, int(0.8*n_samples), replace=False)
        val_indices = np.setdiff1d(np.arange(n_samples), train_indices)
        return train_indices, val_indices
    

    def _compute_composition_scores_df(self,X0,T,Y):
        
        n_samples = X0.shape[0]
        n_measurements = T.shape[1]

        x0_range = self._get_global_x0_range(X0)

        if self.verbose:
            print(f'Fitting the model to {n_samples} samples with {n_measurements} measurements')

        B = self.basis_functions.compute(X0, x0_range)

        X0_tensor, B_tensor, T_tensor, Y_tensor = self._get_tensors(X0,B,T,Y)
        if self.config['fit_single'] == False:
            # Fit all at the same time
            # ------------------------------------------------
            train_indices, val_indices = self._get_train_val_indices(n_samples)

            train_dataset = torch.utils.data.TensorDataset(X0_tensor[train_indices], B_tensor[train_indices], T_tensor[train_indices], Y_tensor[train_indices])
            val_dataset = torch.utils.data.TensorDataset(X0_tensor[val_indices], B_tensor[val_indices], T_tensor[val_indices], Y_tensor[val_indices])

            composition_models = {}
            for composition in self.composition_library:

                composition_config = self.config.copy()
                composition_config['composition'] = composition
                litmodule = LitSketchODE(composition_config)
                val_loss, litmodule = self._fit_composition(composition_config,composition,train_dataset,val_dataset,None,refit=False,tuning=False)
                composition_models[tuple(composition)] = litmodule
            
            composition_scores = {'x0': X0.flatten()}
            for composition, litmodule in composition_models.items():
                litmodule.eval()
                with torch.no_grad():
                    Y_pred = litmodule.model(X0_tensor, B_tensor, T_tensor)
                    errors = torch.mean((Y_pred - Y_tensor) ** 2, dim=1)
                composition_scores[composition] = errors.detach().numpy()
        else:
            # Fit every sample individually
            # ------------------------------------------------
            samples = zip(X0[:,0],T,Y)
            info = zip(samples, [self.composition_library] * n_samples, [self.config['t_range']] * n_samples, [self.config['seed']] * n_samples)
            with multiprocessing.Pool() as p:
                composition_scores = list(tqdm(p.imap(process_sample, info), total=n_samples))
        
        # Fit every sample with the neighbouring samples
        # ------------------------------------------------
        # composition_scores = {'x0':[]}
        # for composition in self.composition_library:
        #     composition_scores[tuple(composition)] = []
        # for i in tqdm(range(n_samples)):
        #     middle = (X0[i][0], T[i], Y[i])
        #     composition_scores['x0'].append(middle[0])
        #     if i > 0 and i < n_samples-1:
        #         # We have a previous and a next sample
        #         before = (X0[i-1][0], T[i-1], Y[i-1])
        #         after = (X0[i+1][0], T[i+1], Y[i+1])
        #         for composition in self.composition_library:
        #             loss, model = calculate_loss_3_samples(composition, self.config['t_range'], before, middle, after, seed=self.config['seed'])
        #             composition_scores[tuple(composition)].append(loss)
        #     else:
        #         for composition in self.composition_library:
        #             loss, model = calculate_loss(composition, self.config['t_range'], middle[0], middle[1], middle[2], seed=self.config['seed'])
        #             composition_scores[tuple(composition)].append(loss)
        
        # Create a dataframe
        composition_scores_df = pd.DataFrame(composition_scores)

        # Sort the dataframe by the column 'x0'
        composition_scores_df = composition_scores_df.sort_values(by='x0', ascending=True) 

        return composition_scores_df
    
    def visualize_composition_scores_df(self, ax):

        if self.composition_scores_df is None:
            raise ValueError('Composition map has not been fitted')
        
        df = self.composition_scores_df

        data_to_plot = np.log(df.iloc[:,1:])

        # Identify outliers in the data by determining the IQR
        Q1 = data_to_plot.quantile(0.25)
        Q3 = data_to_plot.quantile(0.75)
        IQR = Q3 - Q1

        # Filter out the outliers
        data_to_plot[((data_to_plot > (Q3 + 2 * IQR)))] = np.nan

        # Create the heatmap
        sns.heatmap(data_to_plot, ax=ax, annot=False, cmap='viridis', cbar_kws={'shrink': .5})

        # Set x-axis labels to be the column names

        x_labels = [format_composition(column_name) for column_name in df.columns[1:]]
        ax.set_xticks(np.arange(len(x_labels)) + 0.5)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        formatted_y_labels = [f'{label:.2g}' for label in df['x0']]

        step = 2  # Adjust the step to control the number of labels shown
        visible_labels = [label if idx % step == 0 else '' for idx, label in enumerate(formatted_y_labels)]


        ax.set_yticks(np.arange(len(visible_labels)) + 0.5)
        ax.set_yticklabels(visible_labels, rotation=0)

        # Set the aspect ratio to ensure a square-ish plot
        ax.set_aspect('auto')


    def fit_composition_map(self,X0,T,Y):

        if self.verbose:
            print(f'Fitting the composition map to the data')
        
        composition_scores_df = self._compute_composition_scores_df(X0,T,Y)
        self.composition_scores_df = composition_scores_df

        branches = self._solve_branching_problem(composition_scores_df)
        
        composition_map_list = []
        for branch_id in range(len(branches)-1):
            previous_x0, composition = branches[branch_id]
            next_x0, _, = branches[branch_id+1]
            composition_map_list.append(((previous_x0,next_x0), composition))
        last_x0, last_composition = branches[-1]
        composition_map_list.append(((last_x0,np.inf), last_composition))

        composition_map = CompositionMap(composition_map_list)

        if self.verbose:
            print('Composition map:')
            print(composition_map)

        return composition_map
    

    def fit_property_maps(self,X0,T,Y,composition_map):

        if self.verbose:
            print(f'Fitting the property maps to the data')
        
        property_maps = []
        val_loss = 0
        n_val_samples = 0
        torch_models = []

        for i, ((previous_x0, next_x0), composition) in enumerate(composition_map.composition_map_list):

            if self.verbose:
                print(f"Fitting {i+1} out of {len(composition_map)} property maps")
                print(f'Composition {composition} between {previous_x0} and {next_x0}')

            mask = (X0.flatten() > previous_x0) & (X0.flatten() <= next_x0)
            if mask.sum() >= 2:
                # We have points to refit
                
                X0_filtered = X0[mask]
                T_filtered = T[mask]
                Y_filtered = Y[mask]
                x0_range = self._finite_x0_range(previous_x0,next_x0)
                B_filtered = self.basis_functions.compute(X0_filtered,x0_range)

                X0_tensor, B_tensor, T_tensor, Y_tensor = self._get_tensors(X0_filtered,B_filtered,T_filtered,Y_filtered)

                n_samples = X0_filtered.shape[0]

                train_indices, val_indices = self._get_train_val_indices(n_samples)

                train_dataset = torch.utils.data.TensorDataset(X0_tensor[train_indices], B_tensor[train_indices], T_tensor[train_indices], Y_tensor[train_indices])
                val_dataset = torch.utils.data.TensorDataset(X0_tensor[val_indices], B_tensor[val_indices], T_tensor[val_indices], Y_tensor[val_indices])

                composition_config = self.config.copy()
                composition_config['composition'] = composition
                composition_config['refit'] = True
                tuning = (self.config['n_tune'] > 0)
                val_loss_per_branch, model = self._fit_composition(composition_config,composition,train_dataset,val_dataset,i,refit=True,tuning=tuning)
                val_loss += val_loss_per_branch * len(val_indices)
                n_val_samples += len(val_indices)
                property_maps.append(self._construct_single_property_map((previous_x0, next_x0), composition, model.model))
                torch_models.append(model.model)

                if self.verbose:
                    print(f"Validation loss for {i+1} property map: {val_loss_per_branch}")
            
            else:
                raise ValueError('Not enough points to refit the model')
                # In the future you may want to add a default model here
        
        if self.verbose:
            print(f'All property maps fitted')
    
        return property_maps, val_loss/n_val_samples, torch_models


    def fit(self,X0,T,Y,composition_map=None):
        """
        Fit the model to the data

        Args:
        X0: initial condition tensor of shape (batch_size, 1)
        T: time tensor of shape (batch_size, n_measurements)
        Y: output tensor of shape (batch_size, n_measurements)

        Returns:
        """

        self.torch_models = None
        self._get_global_x0_range(X0)

        if composition_map is None:
            composition_map = self.fit_composition_map(X0,T,Y)
        else:
            if self.verbose:
                print('Using the provided composition map')

        property_maps, loss, torch_models = self.fit_property_maps(X0,T,Y,composition_map)
        semantic_predictor = SemanticPredictor(composition_map,property_maps,self.t_range)
        self.semantic_predictor = semantic_predictor
        self.torch_models = torch_models

        print(f"Semantic predictor fitted with validation loss: {loss}")


    def _solve_branching_problem(self, df):

        MIN_BRANCH_LENGTH = 0.1 * (self.x0_range[1] - self.x0_range[0])
        # MIN_SAMPLES = int(0.1 * df.shape[0])
        MIN_SAMPLES = 2

        # create a matrix of Cell objects with the same shape as the input dataframe
        x0 = df['x0'].values
        df = df.drop(columns=['x0'])
        cells = np.empty(df.shape, dtype=object)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                cells[i,j] = Cell()
        
        row_index = cells.shape[0] - 1 # start from the last row
        for j in range(cells.shape[1]):
            cells[row_index,j].total_loss[0] = df.iloc[row_index,j]
            cells[row_index,j].next_composition[0] = None
            cells[row_index,j].samples_without_change[0] = 1
            cells[row_index,j].x0_without_change[0] = 0.0
            for k in range(1,self.max_branches):
                # As it is the last row we cannot have more than 0 changes
                cells[row_index,j].total_loss[k] = np.inf
                cells[row_index,j].next_composition[k] = None
                cells[row_index,j].samples_without_change[k] = 1 # this should never be used
                cells[row_index,j].x0_without_change[k] = 0.0 # this should never be used

        for i in range(1,cells.shape[0]):
            row_index = cells.shape[0] - i - 1
            for j in range(cells.shape[1]):
                # For exactly 0 changes
                cells[row_index,j].total_loss[0] = df.iloc[row_index,j] + cells[row_index+1,j].total_loss[0]
                cells[row_index,j].next_composition[0] = j
                cells[row_index,j].samples_without_change[0] = cells[row_index+1,j].samples_without_change[0] + 1
                cells[row_index,j].x0_without_change[0] = cells[row_index+1,j].x0_without_change[0] + x0[row_index+1] - x0[row_index]
                for k in range(1,self.max_branches):
                    # For exactly k changes
                    min_loss = np.inf
                    min_composition_id = None
                    for l in range(cells.shape[1]):
                        if l == j:
                            # Stay in the same composition
                            if cells[row_index+1,l].total_loss[k] < min_loss:
                                min_loss = cells[row_index+1,l].total_loss[k]
                                min_composition_id = l
                        else:
                            # Switch to a different composition
                            if cells[row_index+1,l].samples_without_change[k-1] < MIN_SAMPLES:
                                # We need at least MIN_SAMPLES samples without change to switch
                                continue
                            if row_index < MIN_SAMPLES - 1:
                                # We are in the first rows, we cannot switch yet (we need MIN_SAMPLES samples without change)
                                continue 
                            if cells[row_index+1,l].x0_without_change[k-1] < MIN_BRANCH_LENGTH:
                                # We need to have a minimum branch length
                                continue
                            if x0[row_index] - x0[0] < MIN_BRANCH_LENGTH:
                                # We need to have a minimum branch length
                                continue
                            if cells[row_index+1,l].total_loss[k-1] < min_loss:
                                min_loss = cells[row_index+1,l].total_loss[k-1]
                                min_composition_id = l
                    cells[row_index,j].total_loss[k] = df.iloc[row_index,j] + min_loss
                    cells[row_index,j].next_composition[k] = min_composition_id
                    cells[row_index,j].samples_without_change[k] = 1 if min_composition_id != j else cells[row_index+1,min_composition_id].samples_without_change[k] + 1
                    cells[row_index,j].x0_without_change[k] = 0.0 if min_composition_id != j else cells[row_index+1,min_composition_id].x0_without_change[k] + x0[row_index+1] - x0[row_index]
        # find the branching with the lowest loss
        branches = []
        min_loss = np.inf
        min_composition_id = None
        min_n_changes = None
        for j in range(cells.shape[1]):
            for k in range(self.max_branches):
                if cells[0,j].total_loss[k] < min_loss:
                    min_loss = cells[0,j].total_loss[k]
                    min_composition_id = j
                    min_n_changes = k
        branches.append((-np.inf, df.columns[min_composition_id]))

        previous_composition_id = min_composition_id
        previous_n_changes = min_n_changes

        for i in range(1,cells.shape[0]):
            next_composition_id = cells[i-1,previous_composition_id].next_composition[previous_n_changes]
            if next_composition_id is None:
                break # we got to the end of the matrix
            if next_composition_id != previous_composition_id:
                # we have a switch
                branches.append((x0[i-1], df.columns[next_composition_id]))
                previous_composition_id = next_composition_id
                previous_n_changes = previous_n_changes - 1
        
        return branches

    def _construct_single_property_map(self,x0_range,composition,torch_model):
        n_motifs = len(composition)
        infinite_composition = (composition[-1][2] != 'c')
        if infinite_composition:
            n_transition_points = n_motifs
        else:
            n_transition_points = n_motifs + 1

        x0_range_finite = self._finite_x0_range(x0_range[0],x0_range[1])

        def create_transition_point_predictor(transition_point_index, coordinate):

            if coordinate == 't':
                coordinate_index = 0
            elif coordinate == 'x':
                coordinate_index = 1

            def transition_point_predictor(X0):

                B = self.basis_functions.compute(X0, x0_range_finite)
                X0_tensor, B_tensor = self._get_tensors(X0,B)

                torch_model.eval()
                with torch.no_grad():
                    coordinates = torch_model.extract_coordinates_finite_composition(X0_tensor, B_tensor).detach().cpu().numpy()

                return coordinates[:,transition_point_index,coordinate_index]
            
            return transition_point_predictor

        def create_derivative_predictor(boundary, order):

            def derivative_predictor(X0):

                B = self.basis_functions.compute(X0, x0_range_finite)
                X0_tensor, B_tensor = self._get_tensors(X0,B)

                torch_model.eval()
                with torch.no_grad():
                    if boundary == 'start':
                        if order == 1:
                            derivative = torch_model.extract_first_derivative_at_start(X0_tensor, B_tensor).detach().cpu().numpy()
                    elif boundary == 'end':
                        if order == 1:
                            derivative = torch_model.extract_first_derivative_at_end(X0_tensor, B_tensor).detach().cpu().numpy()
                        elif order == 2:
                            derivative = torch_model.extract_second_derivative_at_end(X0_tensor, B_tensor).detach().cpu().numpy()
                
                return derivative
        
            return derivative_predictor
        
        def create_infinite_motif_predictor(property_index):

            def infinite_motif_predictor(X0):

                B = self.basis_functions.compute(X0, x0_range_finite)
                X0_tensor, B_tensor = self._get_tensors(X0,B)

                torch_model.eval()
                with torch.no_grad():
                    properties = torch_model.extract_properties_infinite_motif(X0_tensor, B_tensor, self.config['t_range']).detach().cpu().numpy()

                return properties[:,property_index]
            
            return infinite_motif_predictor
        
        transition_point_predictor = {}
        for i in range(n_transition_points):
            for coordinate in ['t','x']:
                transition_point_predictor[(i,coordinate)] = create_transition_point_predictor(i,coordinate)
        
        derivative_predictor = {}
        for boundary in ['start','end']:
            for order in [1,2]:
                if boundary == 'start' and order == 2:
                    continue
                if boundary == 'start' and order == 1 and composition[0][2] != 'c':
                    derivative_predictor[(boundary,order)] = lambda x: np.zeros_like(x)*np.nan
                    continue
                derivative_predictor[(boundary,order)] = create_derivative_predictor(boundary,order)
       
   
        if infinite_composition:
            infinite_motif_predictor = {}
            for i in range(torch_model.number_of_properties_for_infinite_motif()):
                infinite_motif_predictor[i] = create_infinite_motif_predictor(i)
        else:
            infinite_motif_predictor = None

        return SinglePropertyMap(x0_range,x0_range_finite,composition,transition_point_predictor,derivative_predictor,infinite_motif_predictor)

    def predict_raw(self,X0,T):

        if self.semantic_predictor is None:
            raise ValueError('Model has not been fitted yet')
        
        is_x0_scalar = np.isscalar(X0) 
        is_T_scalar = np.isscalar(T)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)
        
        T = np.atleast_1d(T)
        if len(T.shape) == 1:
            T = T.reshape(1,-1)

        semantic_representations = self.semantic_predictor.predict(X0)

        y_pred_list = []
        for i, semantic_representation in enumerate(semantic_representations):
            predictive_model = ApproximatePredictiveModel(semantic_representation)
            y_pred_list.append(predictive_model.forward(T[i]))
        
        results = np.stack(y_pred_list,axis=0)


        if is_x0_scalar:
            results = results[0]
        if is_T_scalar:
            results = results[0]
        
        return results

    
    def predict(self,X0,T):

        is_scalar = np.isscalar(X0)
        is_scalar_T = np.isscalar(T)

        T = np.atleast_1d(T)
        if len(T.shape) == 1:
            T = T.reshape(1,-1)

        X0 = np.atleast_1d(X0)
        if len(X0.shape) == 1:
            X0 = X0.reshape(-1,1)
        

        if self.semantic_predictor is None:
            raise ValueError('Model has not been fitted yet')
        
        semantic_representations = self.semantic_predictor.predict(X0)

        y_pred_list = []
        for i, semantic_representation in enumerate(semantic_representations):
            predictive_model = PredictiveModel(semantic_representation)
            if predictive_model.converged:
                y_pred_list.append(predictive_model.predict(T[i]))
            else:
                y_pred_list.append(self.predict_raw(X0[[i],:],T[[i],:])[0])

        results = np.stack(y_pred_list,axis=0)

        if is_scalar:
            results = results[0]
        if is_scalar_T:
            results = results[0]
        
        return results
       

    def _fit_composition_objective(self,composition_config,composition,train_dataset,val_dataset,property_index,refit,parameters):

        for key, value in parameters.items():
            composition_config[key] = value
        litmodule = LitSketchODE(composition_config)
        assert litmodule.model.composition == composition

        if self.torch_models is not None:
            litmodule.model.fixed_infinite_properties = self.torch_models[property_index].fixed_infinite_properties
            

        composition_timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        if self.verbose:
            print(f'Fitting the model to the data using the composition: {composition}')


        val_loss = np.inf

        # create callbacks
        best_val_checkpoint = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename='best_val',
            dirpath=f'./checkpoints/{composition_timestamp}'
        )
        # added early stopping callback, if validation loss does not improve over 10 epochs -> terminate training.
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10 if not refit else 20,
            verbose=False,
            mode='min'
        )
        callback_ls = [best_val_checkpoint, early_stop_callback]

        trainer_dict = {
            'deterministic': True,
            'devices': 1,
            'enable_model_summary': False,
            'enable_progress_bar': False,
            'accelerator': self.config['device'],
            'max_epochs': self.config['n_epochs'] if not refit else self.config['n_epochs']*5,
            'check_val_every_n_epoch': 2,
            'log_every_n_steps': 1,
            'callbacks': callback_ls
        }
        trainer = L.Trainer(**trainer_dict)

        torch_gen = torch.Generator()
        torch_gen.manual_seed(self.config['seed'])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, generator=torch_gen)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
                
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING) 
        logging.getLogger('lightning').setLevel(0)

            
        trainer.fit(model=litmodule,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
        val_loss = best_val_checkpoint.best_model_score.item()

        # Delete the checkpoint directory
        os.system(f'rm -r ./checkpoints/{composition_timestamp}')

        final_epoch = trainer.current_epoch
        print(f"Finished after {final_epoch} epochs")

        if self.verbose:
            print(f'Validation loss for {composition}: {val_loss}')

        return val_loss


    def _fit_composition(self,composition_config,composition,train_dataset,val_dataset,property_index,refit=False,tuning=False):


        if tuning:
            if self.verbose:
                print(f'Tuning the hyperparameters for the composition: {composition}')
            def objective(trial):
                parameters = {
                    # 'dis_loss_coeff_1': trial.suggest_float('dis_loss_coeff_1', 1e-9, 1e-1, log=True),
                    'dis_loss_coeff_2': trial.suggest_float('dis_loss_coeff_2', 1e-9, 1e-1, log=True),
                    'lr': trial.suggest_float('lr', 1e-4, 1.0, log=True),
                }
                val_loss = self._fit_composition_objective(composition_config,composition,train_dataset,val_dataset,property_index,refit,parameters)
                return val_loss
            
            sampler = optuna.samplers.TPESampler(seed=self.config['seed'])
            study = optuna.create_study(sampler=sampler,direction='minimize')
            study.optimize(objective, n_trials=self.config['n_tune'])
            best_trial = study.best_trial
            best_hyperparameters = best_trial.params
            if self.verbose:
                print(f'Best hyperparameters: {best_hyperparameters}')
            for key, value in best_hyperparameters.items():
                composition_config[key] = value

        litmodule = LitSketchODE(composition_config)
        if self.torch_models is not None:
            litmodule.model.fixed_infinite_properties = self.torch_models[property_index].fixed_infinite_properties

        composition_timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

        if self.verbose:
            print(f'Fitting the model to the data using the composition: {composition}')

        val_loss = np.inf
        num_retries = 0
        while val_loss == np.inf or np.isnan(val_loss):
            if num_retries > 0:
                print(f"Retrying fitting the model with a different seed")
            # create callbacks
            best_val_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                filename='best_val',
                dirpath=f'./checkpoints/{composition_timestamp}'
            )
            # added early stopping callback, if validation loss does not improve over 10 epochs -> terminate training.
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=10 if not refit else 20,
                verbose=False,
                mode='min'
            )
            callback_ls = [best_val_checkpoint, early_stop_callback]

            trainer_dict = {
                'deterministic': True,
                'devices': 1,
                'enable_model_summary': False,
                'enable_progress_bar': False,
                'accelerator': self.config['device'],
                'max_epochs': self.config['n_epochs'] if not refit else self.config['n_epochs']*5,
                'check_val_every_n_epoch': 5,
                'log_every_n_steps': 1,
                'callbacks': callback_ls
            }
            trainer = L.Trainer(**trainer_dict)

            torch_gen = torch.Generator()
            torch_gen.manual_seed(self.config['seed'])

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, generator=torch_gen)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        
                
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING) 
            logging.getLogger('lightning').setLevel(0)

            if num_retries > 0:
                new_config = copy.deepcopy(litmodule.config)
                new_config['seed'] += 1
                litmodule = LitSketchODE(new_config) 
            trainer.fit(model=litmodule,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
            val_loss = best_val_checkpoint.best_model_score.item()
            num_retries += 1

        best_model_path =  best_val_checkpoint.best_model_path
        best_model = LitSketchODE.load_from_checkpoint(checkpoint_path=best_model_path,config=litmodule.config)

        # Delete the checkpoint directory
        os.system(f'rm -r ./checkpoints/{composition_timestamp}')

        final_epoch = trainer.current_epoch
        print(f"Finished after {final_epoch} epochs")
        
        # val_loss = trainer.callback_metrics['val_loss']

        
        if self.verbose:
            print(f'Validation loss for {composition}: {val_loss}')

        return val_loss, best_model
    
    def set_infinite_property_function_through_torch_model(self,property_map_index,infinite_property_index,torch_function):
        
        self.torch_models[property_map_index].fixed_infinite_properties[infinite_property_index] = torch_function

    def refit_property_map(self,property_map_index,X0,T,Y):
       
        if self.semantic_predictor is None:
                raise ValueError('Model has not been fitted yet')

        if self.verbose:
                print(f'Refitting the property map to the data')

        composition_map = self.semantic_predictor.composition_map
        val_loss = 0
       
        previous_x0, next_x0 = composition_map.composition_map_list[property_map_index][0]
        composition = composition_map.composition_map_list[property_map_index][1]

        mask = (X0.flatten() > previous_x0) & (X0.flatten() <= next_x0)
        if mask.sum() >= 2:
            # We have points to refit
            
            X0_filtered = X0[mask]
            T_filtered = T[mask]
            Y_filtered = Y[mask]
            x0_range = self._finite_x0_range(previous_x0,next_x0)
            B_filtered = self.basis_functions.compute(X0_filtered,x0_range)

            X0_tensor, B_tensor, T_tensor, Y_tensor = self._get_tensors(X0_filtered,B_filtered,T_filtered,Y_filtered)

            n_samples = X0_filtered.shape[0]

            train_indices, val_indices = self._get_train_val_indices(n_samples)

            train_dataset = torch.utils.data.TensorDataset(X0_tensor[train_indices], B_tensor[train_indices], T_tensor[train_indices], Y_tensor[train_indices])
            val_dataset = torch.utils.data.TensorDataset(X0_tensor[val_indices], B_tensor[val_indices], T_tensor[val_indices], Y_tensor[val_indices])

            composition_config = self.config.copy()
            composition_config['composition'] = composition
            composition_config['refit'] = True
            tuning = (self.config['n_tune'] > 0)
            val_loss_per_branch, model = self._fit_composition(composition_config,composition,train_dataset,val_dataset,property_map_index,refit=True,tuning=tuning)

            self.semantic_predictor.property_maps[property_map_index] = self._construct_single_property_map((previous_x0, next_x0), composition, model.model)
            self.torch_models[property_map_index] = model.model

            if self.verbose:
                print(f"Property map refitted with validation loss: {val_loss_per_branch}")
        
        else:
            raise ValueError('Not enough points to refit the model')
            # In the future you may want to add a default model here



    def _get_default_opt_config(self):
        config = {
            'device': 'cpu',
            'n_epochs': 1000,
            'batch_size': 256,
            'lr': 1.0e-1,
            'weight_decay': 1e-4,
            'fit_single': True,
        }
        return config
