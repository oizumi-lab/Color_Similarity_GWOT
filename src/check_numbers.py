#%%
import numpy as np
import os
import pandas as pd

def make_symmetric(matrix):
    nrows, ncols = matrix.shape
    if nrows != ncols:
        raise ValueError("Matrix must be square")

    for i in range(nrows):
        for j in range(i+1, ncols):
            if np.isnan(matrix[i, j]) and not np.isnan(matrix[j, i]):
                matrix[i, j] = matrix[j, i]
            elif not np.isnan(matrix[i, j]) and np.isnan(matrix[j, i]):
                matrix[j, i] = matrix[i, j]
            elif not np.isnan(matrix[i, j]) and not np.isnan(matrix[j, i]):
                mean_val = np.mean([matrix[i, j], matrix[j, i]])
                matrix[i, j] = matrix[j, i] = mean_val

    return matrix

# Load the data file
num_response = []
num_non_Nan = []
num_non_Nan_diagonal = []

data_dir = "../data/color_atypical/new_participant_matrices/"
data_list = os.listdir(data_dir)
#for i in range(257):
for fname in data_list:
    #data_dir = "../data/color_atypical/numpy_data/"
    #filepath = f"participant_{i}.npy"
    #data = np.load(data_dir + filepath, allow_pickle=True)
    #data = data.astype(np.float64)
    
    if "csv" not in fname:
        continue
    print(fname)
    data = pd.read_csv(data_dir + fname, header=None)
    data.drop(data.columns[0], axis=1, inplace=True)
    data.drop(data.index[0], axis=0, inplace=True)
    data = data.to_numpy().astype(np.float64)

    # count the number of non NAN elements
    num_non_Nan.append(np.count_nonzero(~np.isnan(data)))
    
    # count the number of diagonal elements
    num_diagonal = np.count_nonzero(~np.isnan(np.diag(data)))
    num_non_Nan_diagonal.append(num_diagonal)
    
    ## replace the diagonal elements with NAN
    #np.fill_diagonal(data, np.nan)
    # check if the matrix is symmetric
    if not np.allclose(data, data.T, equal_nan=True):
        print(f"participant is not symmetric")
        data = make_symmetric(data)
    
    # Get lower triangle indices where data is not NaN
    lower_triangle_indices = np.tril_indices(data.shape[0], -1)  # -1 excludes the diagonal
    values = data[lower_triangle_indices]
    non_nan_indices = np.where(~np.isnan(values))

    # Get final indices and values
    final_indices = (lower_triangle_indices[0][non_nan_indices], lower_triangle_indices[1][non_nan_indices])
    final_values = values[non_nan_indices]

    X = list(zip(*final_indices)) # Zip the indices to get (row, col) pairs
    y = list(final_values)
    
    num_response.append(len(y))
# %%
print(num_non_Nan)
print(num_response)
print(num_non_Nan_diagonal)
print('min : ', min(num_response))
# %%
