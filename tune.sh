# Change according to your environment set-up
source /path/to/venv/bin/activate
cd /path/to/flowssm
export PYTHONPATH=.

directory="/path/to/model/dir" 

# Launch training.
exec python3 scripts/tune_model.py --directory=$directory\
