# Fine-tuning Poseidon on The Well

## 1. Assemble Dataset

Use the original well data

```bash
python assemble_well_active_matter.py \
    --input_dir /data0/arshkon/data/the_well/datasets/active_matter/data/train \
    --output_file /data0/arshkon/data/the_well/datasets/active_matter/assembled_train.nc

python assemble_well_active_matter.py \
    --input_dir /data0/arshkon/data/the_well/datasets/active_matter/data/test \
    --output_file /data0/arshkon/data/the_well/datasets/active_matter/assembled_test.nc

python assemble_well_active_matter.py \
    --input_dir /data0/arshkon/data/the_well/datasets/active_matter/data/valid \
    --output_file /data0/arshkon/data/the_well/datasets/active_matter/assembled_val.nc
```

This assembles .hdf5 files from each split into separate NetCDF files with fields: concentration, velocity, D tensor, E tensor.

## 2. Train Model

```bash
./train_well_active_matter.sh
```

This script:
- Uses config: `configs/well_active_matter_training.yaml`
- Fine-tunes from: `camlab-ethz/Poseidon-B` 
- Saves checkpoints to: `/data0/arshkon/checkpoints/poseidon`

## 3. Evaluate Model

```bash
./eval_well_active_matter.sh
```

Results saved as JSON with metrics like VRMSE, L1 error, and relative error.
