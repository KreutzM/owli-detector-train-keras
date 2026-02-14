# Dataset Runbook (PowerShell)

## Validate COCO

```powershell
python -m owli_train dataset validate --coco tests\data\coco_min.json
```

Optional image-file existence check:

```powershell
python -m owli_train dataset validate --coco data\instances.json --images-dir data\images
```

## Normalize COCO

Normalizes category IDs to deterministic contiguous IDs, optionally merging labels via label map.

```powershell
python -m owli_train dataset normalize --coco data\instances.json --out work\normalized\instances.json
```

With image checks and label mapping:

```powershell
python -m owli_train dataset normalize --coco data\instances.json --images-dir data\images --label-map configs\label_map.yaml --out work\normalized\instances.json
```

## Split COCO (train/val/test)

Writes `splits.json` deterministically by seed:

```powershell
python -m owli_train dataset split --coco data\instances.json --out-dir work\splits --seed 1337
```

Also emit per-split COCO files:

```powershell
python -m owli_train dataset split --coco data\instances.json --out-dir work\splits --seed 1337 --write-coco
```

Produces:
- `work\splits\splits.json`
- `work\splits\instances_train.json`
- `work\splits\instances_val.json`
- `work\splits\instances_test.json`

## Label map format

`configs\label_map.yaml` can be one of these forms:

```yaml
map:
  human: person
  automobile: car
```

```yaml
human: person
automobile: car
```
