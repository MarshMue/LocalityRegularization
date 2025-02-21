#!/bin/bash

# Array of dataset names
datasets=(
    "Salinas"
    "Indian_Pines"
    "Pavia_Centre"
    "Pavia_University"
    "KSC"
    "Botswana"
)

# Run best_rho.py for each dataset sequentially
for dataset in "${datasets[@]}"
do
    echo "Processing dataset: $dataset"
    python best_rho.py --dataset "$dataset"
    echo "Completed processing $dataset"
    echo "------------------------"
done

echo "All datasets processed successfully"