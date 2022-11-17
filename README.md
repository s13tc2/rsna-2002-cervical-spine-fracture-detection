# rsna-2022-cervical-spine-fracture-detection

Reviewing 1st place Kaggle solution from [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/overview) competition

To run DDP training (e.g., `scripts/stage2type1`):
- `cd scripts/stage2type1`
- `torchrun --standalone --nproc_per_node=gpu trainer.py 75 5`

To run DDP training (e.g., `scripts/stage2type2`):
- `cd scripts/stage2typ2`
- `torchrun --standalone --nproc_per_node=gpu trainer.py 50 5`
