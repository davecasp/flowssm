# Change according to your environment set-up
source /path/to/venv/bin/activate
cd /path/to/flowssm
export PYTHONPATH=.

run_name=femur_run
log_dir=runs/$run_name
data_root=/path/to/femur/data

# Create run directory if it doesn't exist.
mkdir -p runs

# adjust lod values accordingly [femur: (1 7), liver: (1 6), classification: (1 10)]
# adjust epsilon values accordingly [femur: 0.53723, liver: 0.43271, classification: 1.43416]
# change data=femur to liver if needed

# Launch training.
exec python3 scripts/train.py \
--loss_type="l1" \
--atol=1e-4 \
--rtol=1e-4 \
--data_root=$data_root \
--lr=1e-3 \
--log_dir=$log_dir \
--lr_scheduler \
--batch_size=16 \
--epochs=300 \
--adjoint \
--solver='dopri5' \
--data='femur' \
--deformer_nf=128 \
--lat_dims=128 \
--nonlin='leakyrelu' \
--lod 1 7 \
--increase_layer=150 \
--global_extent \
--sample_surface \
--epsilon 0.5372271213390913 \
--irregular_grid \
--rbf \
--independent_epsilon \
