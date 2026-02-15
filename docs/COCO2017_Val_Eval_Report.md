# COCO2017 val2017 Evaluation Report (EfficientDet-Lite2)

Status: Template (to be populated by `scripts/eval_coco_val2017.sh`)

## Run Metadata

- Date (UTC): `TBD`
- Branch: `TBD`
- Commit: `TBD`
- COCO annotations: `data/coco2017/annotations/instances_val2017.json`
- COCO images dir: `data/coco2017/val2017`
- Evaluated images: `5000`
- Max detections per image: `100`
- Noise thresholds: `0.05, 0.1, 0.3`

## Models

- Fine-tuned model: `TBD`
- Baseline model: `TBD`

## COCO Metrics (mAP)

| Model | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Fine-tuned | TBD | TBD | TBD | TBD |
| Baseline | TBD | TBD | TBD | TBD |

## Noise Metrics (FP per 100 images)

| Model | @0.05 | @0.10 | @0.30 |
|---|---:|---:|---:|
| Fine-tuned | TBD | TBD | TBD |
| Baseline | TBD | TBD | TBD |

## Per-Class Aggregate (@0.30)

| Model | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Fine-tuned | TBD | TBD | TBD | TBD | TBD |
| Baseline | TBD | TBD | TBD | TBD | TBD |

## Notes

- mAP is computed from detections with `score >= 0.0` (all model outputs retained by TFLite postprocess + max detections cap).
- Noise metrics are reported at fixed thresholds for comparability.
