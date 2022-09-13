# Change according to your environment set-up
source /path/to/venv/bin/activate
cd /path/to/flowssm
export PYTHONPATH=.

directory="/runs/template"

# Lauch hub and spokes method.
exec python3 scripts/generate_template.py \
--log_dir=$directory \
--data_root="/path/to/data" \
--data="femur" \
