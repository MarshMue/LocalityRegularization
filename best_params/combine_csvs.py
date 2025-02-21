import pandas as pd
import glob
import os

def combine_csv_files(input_pattern='*.csv', output_file='combined_results.csv'):
    """
    Combines all CSV files matching the input pattern into a single CSV file.
    Adds a column indicating the source file name.
    """
    # Get list of all CSV files
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print("No CSV files found matching the pattern!")
        return
    
    # Create empty list to store dataframes
    dfs = []
    
    # Read each CSV file
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add filename as a column (without .csv extension)
            df['source_file'] = os.path.splitext(os.path.basename(file))[0]
            dfs.append(df)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    if not dfs:
        print("No data frames to combine!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataframe
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully combined {len(dfs)} files into {output_file}")
    print(f"Total rows: {len(combined_df)}")

if __name__ == "__main__":
    combine_csv_files()