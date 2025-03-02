import numpy as np
import matplotlib.pyplot as plt
from semantic_odes.infinite_motifs import get_default_motif_class
from semantic_odes import utils

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
        fig.suptitle(f'Property map for composition {utils.format_composition(self.composition)}',fontsize=16)

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