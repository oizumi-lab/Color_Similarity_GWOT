#%%
import random
import numpy as np
import matplotlib.pyplot as plt
import ot
import pandas as pd
from tqdm import tqdm

from GW_methods.src.utils.init_matrix import InitMatrix

#%%
def calc_accuracy_with_topk_diagonal(matrix, k, order="maximum", category_mat=None):
        # Get the diagonal elements
        if category_mat is None:
            diagonal = np.diag(matrix)
        else:
            category_mat = category_mat.values

            diagonal = []
            for i in range(matrix.shape[0]):
                category = category_mat[i]

                matching_rows = np.where(np.all(category_mat == category, axis=1))[0]
                matching_elements = matrix[i, matching_rows] # get the columns of which category are the same as i-th row

                diagonal.append(np.max(matching_elements))

        # Get the top k values for each row
        if order == "maximum":
            topk_values = np.partition(matrix, -k)[:, -k:]
        elif order == "minimum":
            topk_values = np.partition(matrix, k - 1)[:, :k]
        else:
            raise ValueError("Invalid order parameter. Must be 'maximum' or 'minimum'.")

        # Count the number of rows where the diagonal is in the top k values
        #count = np.sum(np.isin(diagonal, topk_values))
        count = np.sum([diagonal[i] in topk_values[i] for i in range(matrix.shape[0])])
        
        # Calculate the accuracy as the proportion of counts to the total number of rows
        accuracy = count / matrix.shape[0]
        accuracy *= 100

        return accuracy

#%%
# Set the source and target sizes
N = 93

# topk
topk_list = [1, 3, 5]

# set the result dataframe to store the accuracy
result_df = pd.DataFrame(columns=["Top-k", "Accuracy"])

# Initialize the InitMatrix class
init_mat = InitMatrix(N, N)

# Generate a random initial matrix
n_samples = 10000
init_mat_plan = "random"

for i in tqdm(range(n_samples)):
    T0 = init_mat.make_initial_T(init_mat_plan, seed=i)
    
    for topk in topk_list:
        accuracy = calc_accuracy_with_topk_diagonal(T0, topk)
        #print(f"Sample {i}, Top-{topk} Accuracy: {accuracy:.2f}%")
        result = {"Top-k": topk, "Accuracy": accuracy}
        
        # concat the result to the dataframe
        result_df = pd.concat([result_df, pd.DataFrame([result])])
# %%
# Calculate the mean and variance of the accuracy for each top-k value
result_df.groupby("Top-k").agg(["mean", "std"])

# print the mean and variance of accuracy
print(result_df.groupby("Top-k").agg(["mean", "std"]))

# print the mean and variance of accuracy
print(result_df.groupby("Top-k").agg(["mean", "std"])["Accuracy"]["std"] ** 2)

# %%
