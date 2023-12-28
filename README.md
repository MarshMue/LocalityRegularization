Code to reproduce the results in the paper "Locality Regularized Reconstruction: Structured Sparsity and Delaunay Triangulations". 

The jupyter notebook `paper_figures.ipynb` contains the code to reproduce the figures. We have included relevant data to replicate results in the paper, in particular we included the scaling data obtained from running the script `scaling_all.py` (large `d` problems will take a long time to solve for some methods - see the paper for expected runtimes and scaling results). 

In order to run the scaling experiments you will also need to download [DelaunaySparse](https://github.com/vtopt/DelaunaySparse) and use their python bindings obtained [here](https://github.com/vtopt/DelaunaySparse/blob/main/python/example.py) in the function `delaunay_simplex`. 