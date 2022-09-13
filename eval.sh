# Change according to your environment set-up
source /path/to/venv/bin/activate
cd /path/to/flowssm
export PYTHONPATH=.

directory="/path/to/model/dir" 

# Launch evaluation.
exec python3 scripts/eval_model.py --directory=$directory \
--test \
