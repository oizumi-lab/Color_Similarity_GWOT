# ColorQualiaAlignment

## How to reproduce the results
0. poetry install 
1. Run data_preprocessing.py for preprocessing data.
2. Run gw_optimization.py for obtaining optimal transportation plans.
   1. Precomputed results are available in /results/optimization
3. Run plot_optimal_transportation_plans.py to make plots of optimal transportation plans between all the pairs.
4. Run make_fig1c.py, make_fig3.py, and make_fig4.py to reproduce the figures in the manuscript.
5. Run make_gif_animation.py to get the supplementary videos of aligned embeddings