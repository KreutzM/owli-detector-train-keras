# Android Deploy: TFLite Builtins vs Select TF Ops (Flex)

## Why this matters

TensorFlow Lite models can run with:
- Builtin TFLite ops only (`builtin_ops_only: true`)
- Builtins + Select TF Ops (Flex) (`builtin_ops_only: false`)

You can check this with:

```powershell
python -m owli_train inspect tflite --model work\runs\<run_id>\artifacts\detector.tflite
```

The export metadata file (`detector.tflite.meta.json`) also contains:
- `android_compat.builtin_ops_only`
- `android_compat.requires_select_tf_ops`
- `android_compat.operator_names`

## Trade-offs

Builtins-only:
- Smaller app/runtime footprint
- Simpler Android integration
- Best default for production/offline deployment

Select TF Ops (Flex):
- Supports a wider set of model ops
- Larger runtime footprint and more complex packaging
- May reduce portability across minimal TFLite runtimes

## Export controls

Default export:

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id>
```

Enforce Builtins-only gate:

```powershell
python -m owli_train export tflite --run-dir work\runs\<run_id> --require-builtins-only
```

If gate fails, try:
- `--quant none` or `--quant fp16`
- Export from `.keras` source (`--model ...\detector.keras`)
- A different detector architecture/backbone

## Builtins-first recommendation

For easiest Android deployment, prefer a RetinaNet smoke/baseline config:

```powershell
python -m owli_train train detect --config configs\train_detector_builtins_smoke.yaml --max-steps 1
python -m owli_train export tflite --run-dir work\runs\<run_id> --require-builtins-only
python -m owli_train inspect tflite --model work\runs\<run_id>\artifacts\detector.tflite
```

Expected inspect output: `builtin_ops_only: true`.

## Android guidance

If `builtin_ops_only: true`:
- Use standard TensorFlow Lite Android runtime.
- No Flex dependency needed.

If `builtin_ops_only: false`:
- Include Select TF Ops (Flex) support in Android packaging.
- Validate startup/inference on-device early (runtime init can fail if Flex support is missing).
- Prefer moving back to Builtins-only for release builds when feasible.
