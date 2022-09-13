# Change according to your environment set-up
source /path/to/venv/bin/activate
cd /path/to/flowssm
export PYTHONPATH=.

directory="/path/to/model/dir" 
gt_data="/path/to/ground_truth_data" 

# Launch training.
exec python3 scripts/evaluate_experiment.py --directory=$directory \
--gt_data=$gt_data \
