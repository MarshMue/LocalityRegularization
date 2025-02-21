Code to reproduce the results in the paper "Locality Regularized Reconstruction: Structured Sparsity and Delaunay Triangulations". 

The jupyter notebook `paper_figures.ipynb` contains the code to reproduce the figures. We have included relevant data to replicate results in the paper, in particular we included the scaling data obtained from running the script `scaling_all.py` (large `d` problems will take a long time to solve for some methods - see the paper for expected runtimes and scaling results). 

Additionally, for the hyperspectral image experiments, the complete replication steps are as follows:
1. Download the hyperspectral data and ground truth from https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes and place them in a directory called 'hsi_datasets'. We use the corrected versions when applicable. 
2. Run `best_rho.sh` and `best_lambda.sh` to populate directroy 'best_params'. 
3. Run `combine_csvs.py` in the 'best_params' directory to combine the new csvs. This will be used to select the best parameter during classification. 
4. Run `classify_all_datasets.py` to populate 'all_results' with accuracy data per each trial. 
5. Run `create_dataset_classification_plots.py` to populate 'dataset_classification' with plots and tables. The plots are not used in the paper, but provide a helpful visualization. 

Note: in the above steps, the populated folders have existing data that was used to create the data in the paper. Since there is randomness, replicated results will differ in subtle ways. The main pattern of results should not change. 

In order to run the scaling experiments you will also need to download [DelaunaySparse](https://github.com/vtopt/DelaunaySparse) and use their python bindings obtained [here](https://github.com/vtopt/DelaunaySparse/blob/main/python/example.py) in the function `delaunay_simplex`. 

The following packages will be required:
- `cvxopt`
- `matplotlib`
- `scipy`
- `numpy`
- `tqdm`
- `einops`