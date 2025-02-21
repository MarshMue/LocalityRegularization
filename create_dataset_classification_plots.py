import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_filename(filename):
    """Extract parameters from the filename."""
    pattern = r'(.+)_n_(\d+)_N_(\d+)_f_([^_]+)rho_([^_]+)_lambda_([^_]+)_'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")
    
    return {
        'dataset': match.group(1),
        'n': int(match.group(2)),
        'N': int(match.group(3)),
        'f': float(match.group(4)),
        'rho': float(match.group(5)),
        'lambda': float(match.group(6))
    }

def process_dataset(directory):
    """Process all CSV files for a dataset."""
    dataset_files = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            try:
                params = parse_filename(filename)
                
                df = pd.read_csv(os.path.join(directory, filename), skipinitialspace=True)
                
                if params['dataset'] not in dataset_files:
                    dataset_files[params['dataset']] = []
                
                dataset_files[params['dataset']].append({
                    'params': params,
                    'data': df
                })
            except Exception as e:
                print(f"Skipping {filename}: {e}")
    
    return dataset_files

def plot_dataset_results(dataset_data):
    """Create plots for each dataset."""
    cmap = plt.cm.plasma
    
    for dataset, files in dataset_data.items():
        plt.figure(figsize=(10, 6))
        
        grouped_files = {}
        for file_info in files:
            key = (file_info['params']['rho'], file_info['params']['lambda'])
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file_info)
        
        # Include all columns now
        plot_columns = files[0]['data'].columns
        num_lines = len(plot_columns)
        
        for i, column in enumerate(plot_columns):
            f_values = []
            column_values = []
            column_stds = []
            
            unique_f_values = sorted(set(file['params']['f'] for file in files))
            
            for f in unique_f_values:
                f_column_values = [
                    file['data'][column] 
                    for file in files 
                    if file['params']['f'] == f
                ]
                
                if f_column_values:
                    f_values.append(f)
                    column_values.append(np.mean(f_column_values[0]))
                    column_stds.append(np.std(f_column_values[0]))
            
            color = cmap(i / (num_lines - 1))
            plt.semilogx(f_values, column_values, label=column, color=color)
            # plt.fill_between(
            #     f_values, 
            #     np.array(column_values) - np.array(column_stds),
            #     np.array(column_values) + np.array(column_stds),
            #     alpha=0.2, 
            #     color=color
            # )
        
        representative_file = files[0]['params']
        plt.title(f"{dataset}, rho={representative_file['rho']}, " + 
                  f"lambda={representative_file['lambda']}, " + 
                  f"N={representative_file['N']}")
        
        plt.xlabel('Fraction of Labeled Data')
        plt.ylim([0,1])
        plt.ylabel('Accuracy')
        plt.legend(title='Metric', loc='best')
        plt.tight_layout()
        
        output_filename = f"dataset_classification/{dataset}_results.png"
        plt.savefig(output_filename)
        plt.close()
        
        print(f"Saved plot for {dataset} to {output_filename}")

def format_fraction(num):
    """Format a number to remove trailing zeros after decimal point."""
    if isinstance(num, str):
        return num
    # Convert scientific notation to decimal format for small numbers
    return f"{num:.10f}".rstrip('0').rstrip('.')

def generate_latex_table(dataset_data):
    """Generate LaTeX table for each dataset with vertical lines and double lines."""
    # Define the ordered list of methods and their formatted headers
    method_order = ['NN', 'l1-residual', 'R-max mass', 'R-residual']
    method_headers = {
        'NN': 'NN',
        'l1-residual': '$\ell_1$-residual',
        'R-max mass': '\eqref{eqn:relaxed}-max mass',
        'R-residual': '\eqref{eqn:relaxed}-residual'
    }
    
    for dataset, files in dataset_data.items():
        unique_f_values = sorted(set(file['params']['f'] for file in files))
        methods = method_order  # Use the ordered list instead of files[0]['data'].columns
        
        # Modified table format to include vertical lines (|) and a double vertical line (||)
        latex_lines = [r"\begin{tabular}{|l||" + "|".join(["c"] * len(methods)) + "|}"]
        
        # line at top
        latex_lines.append(r"\hline")
        
        
        # Header row with formatted method names
        header = [r"$f$"] + [method_headers[method] for method in methods]
        latex_lines.append(" & ".join(header) + r" \\")

        # Double horizontal line after header
        latex_lines.append(r"\hline\hline")
    
        
        # Store data for each fraction and method
        f_data = {f: {method: {'mean': 0, 'std': 0} for method in methods} for f in unique_f_values}
        
        # Collect all values
        for f in unique_f_values:
            for method in methods:
                f_method_values = [
                    file['data'][method]
                    for file in files
                    if file['params']['f'] == f
                ]
                if f_method_values:
                    mean_val = np.mean(f_method_values[0])
                    std_val = np.std(f_method_values[0])
                    f_data[f][method] = {'mean': mean_val, 'std': std_val}
        
        # Create table rows
        for f in unique_f_values:
            row_data = [format_fraction(f)]  # Special formatting only for fraction values
            # Find maximum mean for this row
            row_means = [f_data[f][method]['mean'] for method in methods]
            max_mean = max(row_means)
            
            for method in methods:
                if f_data[f][method]['mean'] != 0:
                    # Use original formatting for mean and std values
                    cell = f"{f_data[f][method]['mean']:.3f} ({f_data[f][method]['std']:.3f})"
                    # Bold if this is the highest value in the row
                    if f_data[f][method]['mean'] == max_mean:
                        cell = r"\textbf{" + cell + "}"
                    row_data.append(cell)
                else:
                    row_data.append("-")
            
            latex_lines.append(" & ".join(row_data) + r" \\")
            latex_lines.append(r"\hline")
        
        latex_lines.append(r"\end{tabular}")
        
        with open(f"dataset_classification/{dataset}_results_table.tex", "w") as f:
            f.write("\n".join(latex_lines))
        print(f"Saved LaTeX table for {dataset} to {dataset}_results_table.tex")

def main(directory):
    """Main function to process, plot, and generate LaTeX table."""
    dataset_data = process_dataset(directory)
    plot_dataset_results(dataset_data)
    generate_latex_table(dataset_data)

# Example usage
# main('/path/to/your/csv/directory')

if __name__ == "__main__":
    main("/Users/marshallmueller/PycharmProjects/LocalityRegularization/all_results_new_lambda/")