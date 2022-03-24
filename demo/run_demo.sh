#!/bin/bash
echo "Running Neural ADMIXTURE on demo data..."
start=`date +%s`
neural-admixture train --k 7 --data_path data/demo_data.bed --save_dir outputs --name demo_run --max_epochs 5 --seed 42
end=`date +%s`
runtime=$((end-start))
echo "Demo run in ${runtime} seconds."
echo "Running diagnostics..."
python3 run_diagnostics.py
