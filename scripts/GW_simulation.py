#%%
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../../'))

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr

from GW_methods.src.align_representations import Representation, AlignRepresentations, OptimizationConfig, VisualizationConfig
#%%
# load colors
colors = np.load("../data/hex_code/new_color_order.npy")

### Simulation for the colors data number 3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_random_points(n, N, spread=1.0, seed=None):
    """
    Generates N random points in n-dimensional space with a specified spread.
    
    Args:
    n (int): Number of dimensions.
    N (int): Number of points to generate.
    spread (float): The spread factor determining the range of the points. Default is 1.0.
    
    Returns:
    numpy.ndarray: An array of shape (N, n) containing N points in n-dimensional space.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(N, n) * spread

def generate_points_around(points, m, spread=0.1, seed=None):
    """
    Generates m random points around each of the given N points, returning a total of N*m points.
    
    Args:
    points (numpy.ndarray): An array of shape (N, n) containing N points in n-dimensional space.
    m (int): Number of points to generate around each given point.
    spread (float): The spread factor determining the range around each point. Default is 0.1.
    
    Returns:
    numpy.ndarray: An array of shape (N*m, n) containing N*m points in n-dimensional space.
    """
    if seed is not None:
        np.random.seed(seed)
    N, n = points.shape
    generated_points = []

    for point in points:
        # Generate m random points around each given point
        new_points = point + (np.random.rand(m, n) - 0.5) * 2 * spread
        generated_points.append(new_points)

    # Combine all new points into a single array
    return np.vstack(generated_points)

def plot_points_in_2d(points):
    """
    Reduces the dimensionality of the given points to 2D using PCA and plots them.
    
    Args:
    points (numpy.ndarray): An array of shape (N, n) containing N points in n-dimensional space.
    """
    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points)
    
    # Plot the 2D points
    plt.figure(figsize=(8, 8))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], alpha=0.7)
    plt.title("2D Projection of Points")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.show()

def get_RDM(embedding):
    """
    Computes the representational dissimilarity matrix (RDM) for the given embedding.
    
    Args:
    embedding (numpy.ndarray): An array of shape (N, n) containing N points in n-dimensional space.
    
    Returns:
    numpy.ndarray: The RDM for the given embedding.
    """
    return distance.cdist(embedding, embedding, metric="euclidean")

def RSA(matrix1, matrix2):
    """
    Computes the correlation between the upper triangles of the two given matrices.
    
    Args:
    matrix1 (numpy.ndarray): The first matrix.
    matrix2 (numpy.ndarray): The second matrix.
    
    Returns:
    float: The correlation between the upper triangles of the two matrices.
    """
    # Get the upper triangles of the matrices
    upper_triangle1 = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
    upper_triangle2 = matrix2[np.triu_indices(matrix2.shape[0], k=1)]
    
    # Compute the correlation between the upper triangles
    return pearsonr(upper_triangle1, upper_triangle2)[0]

#%%
# Search for the optimal noise level that produces the desired correlations and accuracies
n_dimensions = 30
n_points_per_clusters = [2, 5]#

# objective accuracy
objective_accuracis = [50, 18]#

spread_centers = 10 # fix

iter_max = 1000

for objective, n_points_per_cluster in zip(objective_accuracis, n_points_per_clusters):
    
    seed_log = []
    spread_around_log = []
    acc_log = []
    
    acc = 0
    n_iter = 0
    pre_seed_list = []
    
    # iterate until the accuracy is above the objective accuracy
    while acc < objective and n_iter < iter_max:
        # Cluster size
        n_clusters = 93 // n_points_per_cluster
        interpolation_points = 93 - n_clusters * n_points_per_cluster

        # Generate random points around the centers
        correlation = 1.0
        spread_around = 0.1 # noise

        # keep the correlation around 0.6
        while correlation > 0.7:
            # set seed randomly
            seed = np.random.randint(10000)
            
            # check if the seed is already used
            while seed in pre_seed_list:
                seed = np.random.randint(10000)
                
            # Generate the points
            centers = generate_random_points(n_dimensions, n_clusters, spread=spread_centers, seed=seed)
            embedding_1 = generate_points_around(centers, n_points_per_cluster, spread=spread_around, seed=seed+1)
            embedding_2 = generate_points_around(centers, n_points_per_cluster, spread=spread_around, seed=seed+2)

            if interpolation_points > 0:
                interp = generate_random_points(n_dimensions, interpolation_points, spread=spread_centers)
                embedding_1 = np.vstack((embedding_1, interp))
                embedding_2 = np.vstack((embedding_2, interp))

            # Compute the RDMs
            RDM_1 = get_RDM(embedding_1)
            RDM_2 = get_RDM(embedding_2)

            # Compute the RSA correlation
            correlation = RSA(RDM_1, RDM_2)

            # keep the correlation above 0.6
            if correlation < 0.6:
                spread_around -= 0.1
            else:
                spread_around += 0.1

        # save the seed and the spread
        print(f"n_points_per_cluster: {n_points_per_cluster}, seed: {seed}, spread_around: {spread_around}")
        seed_log.append(seed)
        spread_around_log.append(spread_around)
        
        pre_seed_list.append(seed)
        
        #spread_around = 2
        #centers = generate_random_points(n_dimensions, n_clusters, spread=spread_centers, seed=3)
        #embedding_1 = generate_points_around(centers, n_points_per_cluster, spread=spread_around, seed=1)
        #embedding_2 = generate_points_around(centers, n_points_per_cluster, spread=spread_around, seed=2)
        #plot_points_in_2d(embedding_1)
        #plot_points_in_2d(embedding_2)
        #
        #if interpolation_points > 0:
        #    interp = generate_random_points(n_dimensions, interpolation_points, spread=spread_centers)
        #    embedding_1 = np.vstack((embedding_1, interp))
        #    embedding_2 = np.vstack((embedding_2, interp))

        # alignment
        Group1 = Representation(
            name="1",
            metric="euclidean",
            embedding=embedding_1,
            )

        Group2 = Representation(
            name="2",
            metric="euclidean",
            embedding=embedding_2,

        )

        config = OptimizationConfig(
            eps_list=[0.1, 10],
            num_trial=20, #50
            db_params={"drivername": "sqlite"},
            n_iter=1,
        )

        vis_config = VisualizationConfig(
            figsize=(8, 6), 
            title_size = 0, 
            cmap = "rocket_r",
            cbar_ticks_size=10,
            font="Arial",
            color_labels=colors,
            color_label_width=3
        )

        vis_emb = VisualizationConfig(
            figsize=(8, 8), 
            legend_size=12,
            marker_size=60,
            color_labels=colors,
            fig_ext='svg',
            markers_list=['o', 'X']
        )
        vis_emb_2 = VisualizationConfig(
            figsize=(8, 8), 
            legend_size=12,
            marker_size=60,
            color_labels=colors,
            fig_ext='svg',
            markers_list=['X']
        )

        vis_log = VisualizationConfig(
            figsize=(8, 6), 
            title_size = 0, 
            cmap = "viridis",
            cbar_ticks_size=15,
            font="Arial",
            xlabel_size=20,
            xticks_size=15,
            ylabel_size=20,
            yticks_size=15,
            cbar_label_size=15,
            plot_eps_log=True,
            fig_ext='svg'
        )

        alignment = AlignRepresentations(
            config=config,
            representations_list=[Group1, Group2],
            main_results_dir="../results/",
            data_name=f"Simulation_colors_cluster_{n_clusters}"
        )


        # RSA
        fig_dir = f"../results/figs/Simulation_colors_cluster/{n_clusters}clusters"
        os.makedirs(fig_dir, exist_ok=True)
        alignment.show_sim_mat(
            visualization_config=vis_config, 
            show_distribution=False,
            fig_dir=fig_dir
            )
        alignment.RSA_get_corr()

        # show embeddings
        Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir=fig_dir, fig_name="Group1", legend=False)
        Group2.show_embedding(dim=2, visualization_config=vis_emb_2, fig_dir=fig_dir, fig_name="Group2", legend=False)

        # GW
        alignment.gw_alignment(
            compute_OT=False,
            delete_results=True,
            visualization_config=vis_config,
            fig_dir=fig_dir
            )

        alignment.show_optimization_log(
            fig_dir=fig_dir,
            visualization_config=vis_log
            )
        #alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir=fig_dir)

        df = alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan", return_dataframe=True)

        # check the accuracy
        acc = df.iloc[0].values.item()
        acc_log.append(acc)
        
        n_iter += 1
        
    # save the seed and the spread and the accuracy
    result = pd.DataFrame({
        "seed": seed_log,
        "spread_around": spread_around_log,
        "accuracy": acc_log
    })
    result.to_csv(f"../results/Simulation_colors_cluster_{n_clusters}_results.csv")
# %%


#######
# archived version
#######

use_archived_version = False

if use_archived_version:
    ### Pattern 1
    # generate two circles with different radii and number of points

    def plot_circles_and_points(radius1, num_points1, radius2, num_points2, distance):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')


        # Circle 1
        theta1 = np.linspace(0, 2 * np.pi, num_points1, endpoint=False)
        x1 = radius1 * np.cos(theta1) - distance / 2
        y1 = radius1 * np.sin(theta1)
        #circle1 = plt.Circle((-distance / 2, 0), radius1, edgecolor='b', facecolor='none')
        circle_1 = np.column_stack((x1, y1))

        # Circle 2
        theta2 = np.linspace(0, 2 * np.pi, num_points2, endpoint=False)
        x2 = radius2 * np.cos(theta2) + distance / 2
        y2 = radius2 * np.sin(theta2)
        #circle2 = plt.Circle((distance / 2, 0), radius2, edgecolor='r', facecolor='none')
        circle_2 = np.column_stack((x2, y2))

        #ax.add_patch(circle1)
        #ax.add_patch(circle2)
        ax.plot(x1, y1, 'bo')  # Points for circle 1
        ax.plot(x2, y2, 'ro')  # Points for circle 2

        ax.set_xlim(-radius1 - distance, radius2 + distance)
        ax.set_ylim(-max(radius1, radius2) - 1, max(radius1, radius2) + 1)

        plt.grid(True)
        plt.show()

        return circle_1, circle_2

    # Parameters
    radius1 = 4
    num_points1 = 60
    radius2 = 3
    num_points2 = 33
    distance = 3.5

    circle1, circle2 = plot_circles_and_points(radius1, num_points1, radius2, num_points2, distance)

    # %%
    # concatenate circles
    embedding_1 = np.concatenate((circle1, circle2), axis=0)
    # add noise
    np.random.seed(0)
    noise = np.random.normal(0, 0.3, embedding_1.shape)
    embedding_1 = embedding_1 + noise

    # shuffle the order of the points of the second circle and save the new embedding
    np.random.seed(0)
    np.random.shuffle(circle2)
    embedding_2 = np.concatenate((circle1, circle2), axis=0)
    # add the same noise
    embedding_2 = embedding_2 + noise

    # %%
    Group1 = Representation(
        name="1",
        metric="euclidean",
        embedding=embedding_1,
        )

    Group2 = Representation(
        name="2",
        metric="euclidean",
        embedding=embedding_2,

    )

    config = OptimizationConfig(
        eps_list=[0.1, 1],
        num_trial=10, #50
        db_params={"drivername": "sqlite"},
        n_iter=1,
    )

    vis_config = VisualizationConfig(
        figsize=(8, 6), 
        title_size = 0, 
        cmap = "rocket_r",
        cbar_ticks_size=10,
        font="Arial",
        color_labels=colors,
        color_label_width=3
    )

    vis_emb = VisualizationConfig(
        figsize=(8, 8), 
        legend_size=12,
        marker_size=60,
        color_labels=colors,
        fig_ext='svg',
        markers_list=['o', 'X']
    )
    vis_emb_2 = VisualizationConfig(
        figsize=(8, 8), 
        legend_size=12,
        marker_size=60,
        color_labels=colors,
        fig_ext='svg',
        markers_list=['X']
    )

    vis_log = VisualizationConfig(
        figsize=(8, 6), 
        title_size = 0, 
        cmap = "viridis",
        cbar_ticks_size=15,
        font="Arial",
        xlabel_size=20,
        xticks_size=15,
        ylabel_size=20,
        yticks_size=15,
        cbar_label_size=15,
        plot_eps_log=True,
        fig_ext='svg'
    )

    alignment = AlignRepresentations(
        config=config,
        representations_list=[Group1, Group2],
        main_results_dir="../results/",
        data_name="Simulation_colors"
    )

    #%%
    # RSA
    alignment.show_sim_mat(
        visualization_config=vis_config, 
        show_distribution=False,
        fig_dir="../results/Simulation_colors"
        )
    alignment.RSA_get_corr()

    # show embeddings
    Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors", fig_name="Group1", legend=False)
    Group2.show_embedding(dim=2, visualization_config=vis_emb_2, fig_dir="../results/Simulation_colors", fig_name="Group2", legend=False)
    # %%
    # GW
    alignment.gw_alignment(
        compute_OT=True,
        delete_results=False,
        visualization_config=vis_config
        )

    alignment.show_optimization_log(
        fig_dir="../results/Simulation_colors",
        visualization_config=vis_log
        )
    alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors")

    alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
    # %%



    ### Simulation for the colors data number 2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # load colors
    colors = np.load("../../data/color/new_color_order.npy")

    def plot_circle(radius, num_points):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        circle = np.column_stack((x, y))

        ax.plot(x, y, 'bo')  # Points for circle

        ax.set_xlim(-radius - 1, radius + 1)
        ax.set_ylim(-radius - 1, radius + 1)

        plt.grid(True)
        plt.show()

        return circle

    # Parameters
    radius = 4
    num_points = 93

    circle1 = plot_circle(radius, num_points)
    circle2 = plot_circle(radius, num_points)

    # add correlated noise
    np.random.seed(0)
    noise = np.random.normal(0, 0.4, circle1.shape)
    circle1 = circle1 + noise*1
    circle2 = circle2 + noise*1

    # shuffle the order of the points of the second circle and save the new embedding
    np.random.seed(0)
    np.random.shuffle(circle2[-38:])

    embedding_1 = circle1
    embedding_2 = circle2
    # %%
    # %%
    Group1 = Representation(
        name="1",
        metric="euclidean",
        embedding=embedding_1,
        )

    Group2 = Representation(
        name="2",
        metric="euclidean",
        embedding=embedding_2,

    )

    config = OptimizationConfig(
        eps_list=[0.01, 0.1],
        num_trial=10, #50
        db_params={"drivername": "sqlite"},
        n_iter=1,
    )

    vis_config = VisualizationConfig(
        figsize=(8, 6), 
        title_size = 0, 
        cmap = "rocket_r",
        cbar_ticks_size=10,
        font="Arial",
        color_labels=colors,
        color_label_width=3
    )

    vis_emb = VisualizationConfig(
        figsize=(8, 8), 
        legend_size=12,
        marker_size=60,
        color_labels=colors,
        fig_ext='svg',
        markers_list=['o', 'X']
    )
    vis_emb_2 = VisualizationConfig(
        figsize=(8, 8), 
        legend_size=12,
        marker_size=60,
        color_labels=colors,
        fig_ext='svg',
        markers_list=['X']
    )

    vis_log = VisualizationConfig(
        figsize=(8, 6), 
        title_size = 0, 
        cmap = "viridis",
        cbar_ticks_size=15,
        font="Arial",
        xlabel_size=20,
        xticks_size=15,
        ylabel_size=20,
        yticks_size=15,
        cbar_label_size=15,
        plot_eps_log=True,
        fig_ext='svg'
    )

    alignment = AlignRepresentations(
        config=config,
        representations_list=[Group1, Group2],
        main_results_dir="../results/",
        data_name="Simulation_colors"
    )

    #%%
    # RSA
    alignment.show_sim_mat(
        visualization_config=vis_config, 
        show_distribution=False,
        fig_dir="../results/Simulation_colors"
        )
    alignment.RSA_get_corr()

    # show embeddings
    Group1.show_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors", fig_name="Group1", legend=False)
    Group2.show_embedding(dim=2, visualization_config=vis_emb_2, fig_dir="../results/Simulation_colors", fig_name="Group2", legend=False)
    # %%
    # GW
    alignment.gw_alignment(
        compute_OT=True,
        delete_results=True,
        visualization_config=vis_config
        )

    alignment.show_optimization_log(
        fig_dir="../results/Simulation_colors",
        visualization_config=vis_log
        )
    alignment.visualize_embedding(dim=2, visualization_config=vis_emb, fig_dir="../results/Simulation_colors")

    alignment.calc_accuracy(top_k_list=[1, 3, 5], eval_type="ot_plan")
    # %%
