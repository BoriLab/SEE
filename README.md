# SEE
spectral edge encoding
**Data Preprocessing**.  

Place the following three scripts in the `data/moleculenet_data/` directory:

* `moleculenet_2d.py`
  – Default preprocessing. This is used for the main experiments.
* `moleculenet_big_data_2d.py`
  – Alternative preprocessing for environments with limited storage.
* `moleculenet_2d_non_geo.py`
  – Preprocessing for non-geometric 2D graphs. SEE supports both geometric 2D and non-geometric (0/1) graph representations.

---

**Usage**

Run training with default hyperparameters and the BBBP dataset by executing:

```bash
python train.py \
  --DATASET BBBP \
  --LEARNING_RATE 0.0005 \
  --BATCH_SIZE 64 \
  --DEPTH 4 \
  --MLP_DIM 256 \
  --HEADS 12 \
  --T_MAX 200 \
  --WEIGHT_DECAY 1e-5 \
  --SCALE_MIN 0.5 \
  --SCALE_MAX 1.5 \
  --device cuda \
  --VERBOSE True
```

Alternatively, you can edit the `CONFIG` dictionary directly in the code and then run:

```bash
python train.py
```

Adjust any of the command‐line arguments to suit your needs.
