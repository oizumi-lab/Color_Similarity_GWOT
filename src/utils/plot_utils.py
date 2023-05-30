"""
Some plotting functions

Genji Kawakita
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns



def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def plot_MDS_embedding(embedding,marker,colors,alpha=1,s=200,fig_size=(15,15),save=False,fig_dir=None,save_name=None,view_init=None):
    fig_size = fig_size
    #X_col_dict = out["X_col_dict"]
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs=embedding[:,0],ys=embedding[:,1],zs=embedding[:,2],\
                marker=marker,color=colors,alpha=alpha,s=s)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_zaxis().set_visible(False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    if view_init:
        ax.view_init(elev=view_init[0], azim=view_init[1])
    plt.tight_layout()
    if save:
        plt.savefig(f'{fig_dir}/{save_name}',transparent=True,dpi=300)
    plt.show()
