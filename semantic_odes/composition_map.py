import numpy as np
from semantic_odes import utils


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
            utils.assign_to_mask(mask.flatten(),compositions,composition)
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

class Cell:

    def __init__(self):
        self.next_composition = {}
        self.total_loss = {}
        self.samples_without_change = {}
        self.x0_without_change = {}

def solve_branching_problem(df, x0_range, max_branches):

        MIN_BRANCH_LENGTH = 0.1 * (x0_range[1] - x0_range[0])
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
            for k in range(1,max_branches):
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
                for k in range(1,max_branches):
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
            for k in range(max_branches):
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