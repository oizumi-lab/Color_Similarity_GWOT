"""
Some plotting functions

Genji Kawakita
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation


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



def gif_animation(embedding_list,markers_list,colors,fig_size=(15,12),save_anim=False,save_path=None):
    
    X = embedding_list[0] # referenced embeddings

    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax = fig.add_subplot(111, projection="3d")
    #ax.set_axis_off() 
    n_groups = len(embedding_list)
    for grp_idx in range(n_groups):
        ax.scatter(xs=embedding_list[grp_idx][:,0],ys=embedding_list[grp_idx][:,1],zs=embedding_list[grp_idx][:,2],\
                marker=markers_list[grp_idx],color=colors,alpha=1,s=100,label=f"Group {grp_idx+1}")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xlim()
    ax.set_ylim()
    ax.set_zlim()
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)
    ax.axes.get_zaxis().set_visible(True)
    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')
    ax.set_facecolor('white')
    ax.grid(True)
    #bbox = fig.bbox_inches.from_bounds(1, 1, 8, 6)
    #plt.tight_layout()

    # Create the animation
    def update(frame):
        ax.view_init(elev=10,azim=frame*1)

    anim = FuncAnimation(fig, update, frames=range(180), repeat=False,interval=150)
    
    # Save the animation as a gif
    if save_anim:
        if 'mp4' in save_path:
            anim.save(save_path, dpi=80, writer="ffmpeg")
        if 'gif' in save_path:
            anim.save(save_path, dpi=80, writer="pillow")
        
    # Show the animation
    plt.show()