# Change according to your environment set-up
source /path/to/venv/bin/activate
cd /path/to/flowssm
export PYTHONPATH=.

directory="/path/to/model/dir"

# Launch training.
exec python3 scripts/reconstruction.py \
--directory=$directory \
--recon_dir="/path/to/reconstruction/data" \