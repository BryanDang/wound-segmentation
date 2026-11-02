# Ablation 4d production configuration

Created: 2025-10-21.
Status: configured for production, not deployed (awaiting go-ahead).

## Notebook settings

- File: `ablation_4d_curriculum_learning.ipynb`
- Mode: production (`SMOKE_TEST = False`)
- Device: GPU (`device=0`)
- Seeds: `[42, 123, 456]`

## Training configuration

Epochs per strategy:
- Total: 100 (33 + 33 + 34)
- Phase 1: 33 epochs (easy curriculum)
- Phase 2: 33 epochs (easy and medium curriculum)
- Phase 3: 34 epochs (all data)
- Transitions at epochs 33 and 66

Hyperparameters (matching Ablation 4a):
- `batch=32`
- `lr0=0.015`
- `workers=8`
- `imgsz=512`
- No augmentation
- P100 GPU optimized

## Strategies

### 4d.2: progressive hard negatives

Curriculum on DDI images only. Wounds stay fixed at all data.
- Phase 1: all wounds + easy DDI
- Phase 2: all wounds + easy and medium DDI
- Phase 3: all wounds + all DDI

### 4d.3: bilateral curriculum

Curriculum on both wounds and DDI.
- Phase 1: easy wounds + easy DDI
- Phase 2: easy and medium wounds + easy and medium DDI
- Phase 3: all wounds + all DDI

## Kaggle metadata

```json
{
  "id": "bkdkaggle/ablation-4d-curriculum-production",
  "title": "Ablation 4d: Curriculum Learning Production",
  "enable_gpu": "true",
  "enable_internet": "true"
}
```

Datasets required:
1. `bkdkaggle/baseline-outputs`: baseline model and wound images
2. `bkdkaggle/ddi-diverse-dermatology`: hard negative images

## Expected runtime

Per strategy:
- 100 epochs x 3 seeds = 300 training epochs
- About 3 to 4 hours per seed on P100
- About 10 to 12 hours per strategy

Both strategies: about 20 to 24 hours of GPU time.

## Expected outputs

Each seed produces:
- `ablation_4d2_best.pt` (strategy 4d.2)
- `ablation_4d3_best.pt` (strategy 4d.3)
- `hardness_stratification.json`
- `completion_status.json`
- Training logs and metrics

## Smoke test vs production

| Setting | Smoke test | Production |
|---------|------------|------------|
| SMOKE_TEST | True | False |
| Device | CPU | GPU (device=0) |
| Epochs phase 1 | 3 | 33 |
| Epochs phase 2 | 3 | 33 |
| Epochs phase 3 | 3 | 34 |
| Seeds | [42] | [42, 123, 456] |
| Image limits | 3-10 images | full datasets |
| Total runtime | 5-10 min | 20-24 hours |

## Configuration state

- SMOKE_TEST set to False: done
- Device set to GPU (device=0): done
- Epochs set to 33/33/34 per phase: done
- Multi-seed [42, 123, 456]: done
- Image limits removed (limit=None): done
- Kernel metadata updated (GPU enabled): done
- Title updated to "Production": done
- Kernel ID updated: done

## Deployment command

```bash
# From project root:
cd notebooks/03_ablations/ablation_4d_production
kaggle kernels push
```

## Next

1. Deploy to Kaggle (go-ahead pending).
2. Monitor GPU hours usage.
3. Download results when complete.
4. Analyze results.
5. Compare against Ablation 4a (87.51% mAP50).
