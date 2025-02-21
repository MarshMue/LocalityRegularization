import pandas as pd
import subprocess

# Read the CSV file
df = pd.read_csv('best_params/combined_results.csv')

# Process each unique dataset
for dataset in df['Dataset'].unique():
    # Get the best rho (maximum Best_Param for R and R_res)
    rho_df = df[df['Dataset'] == dataset]
    rho_df = rho_df[rho_df['Method'].isin(['R', 'R_res'])]
    rho = float(rho_df['Best_Param'].max())

    # Get lambda1 (Best_Param for l1_res)
    lambda1_df = df[(df['Dataset'] == dataset) & (df['Method'] == 'l1_res')]
    if not lambda1_df.empty:
        lambda1 = float(lambda1_df['Best_Param'].iloc[0])
    else:
        print(f"Warning: No l1_res method found for {dataset}, skipping...")
        continue

    for frac in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        # Get lambda1 (Best_Param for l1_res)
        lambda1_df = df[(df['Dataset'] == dataset) & (df['Method'] == 'l1_res') & (df['Frac_Labeled'] == frac)]
        if not lambda1_df.empty:
            lambda1 = float(lambda1_df['Best_Param'].iloc[0])
        else:
            print(f"Warning: No l1_res method found for {dataset}, skipping...")
            continue
        # Construct and run the command
        cmd = [
            'python', 'classification_script.py',
            '--rho', str(rho),
            '--lambda1', str(lambda1),
            '--dataset', dataset,
            '--frac_labeled', str(frac)
        ]

        print(f"\nRunning classification for {dataset}:")
        print(f"rho = {rho}, lambda1 = {lambda1}")
        print("Command:", ' '.join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running classification for {dataset}: {e}")
