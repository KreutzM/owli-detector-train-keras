# Codex Task Report

## Ziel
- Den aktuellen Branch `feat/2nd_Et-Lite-2-Train` gegen den angekuendigten PR-Inhalt `stage4 coco replay pipeline` abgleichen.
- Sichtbar machen, dass die Stage-4-Artefakte bereits real im Branch vorhanden sind.
- Den Review-Stand mit einer klaren Stage-4-Doku und einem dazu passenden Pflicht-Report reparieren.

## Was wurde geändert?
- Neue sichtbare Stage-4-Uebersicht:
  - `docs/BA_MVP_Stage4_Replay_Pipeline.md`
- Knappe Verweise auf die neue Stage-4-Uebersicht ergaenzt:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Reparaturtask aktualisiert

## Was wurde wirklich verifiziert?
- Realer Branch-Inhalt und letzter Commit geprueft:
```bash
git branch --show-current
git rev-parse --short HEAD
git --no-pager log --oneline -5
```
  - Exit-Code: `0`
  - Ergebnis:
    - Branch: `feat/2nd_Et-Lite-2-Train`
    - Head: `ee21838`
    - letzter Commit: `feat: add stage4 coco replay pipeline`

- Reale Stage-4-Dateien im Branch geprueft:
```bash
rg -n "stage4|coco replay|coco_replay" configs docs src tests -S --glob '!work/**' --glob '!data/**'
```
  - Exit-Code: `0`
  - Ergebnis:
    - Stage-4-Config vorhanden:
      - `configs/coco_replay_ba_mvp_stage4.yaml`
    - Stage-4-Merge-Manifest vorhanden:
      - `configs/merge_ba_mvp_stage4_with_coco_replay.yaml`
    - naechste Trainingsconfig vorhanden:
      - `configs/efficientdet_lite2_ba_mvp_stage4.yaml`
    - Replay-Importer und Tests vorhanden:
      - `src/owli_train/data/coco_replay.py`
      - `tests/test_dataset_import_coco_replay.py`

- Reale sichtbare Stage-4-Artefakte auf Disk geprueft:
```bash
find work/datasets/coco_replay_ba_v1_stage4 -maxdepth 2 | sort
find work/datasets/ba_mvp_stage4_with_coco_replay -maxdepth 2 | sort | sed -n '1,120p'
find work/splits/ba_mvp_stage4_with_coco_replay -maxdepth 1 | sort
```
  - Exit-Code: `0`
  - Ergebnis:
    - Replay-Artefakte vorhanden:
      - `work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json`
      - `work/datasets/coco_replay_ba_v1_stage4/class_names.json`
      - `work/datasets/coco_replay_ba_v1_stage4/qc_report.json`
    - Stage-4-Merge-Artefakte vorhanden:
      - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json`
      - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.report.json`
      - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json`
      - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
    - Stage-4-Split-Artefakte vorhanden:
      - `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`
      - `instances_train.json`
      - `instances_val.json`
      - `instances_test.json`

- Inhaltliche Stage-4-Sichtbarkeit in den zentralen Docs geprueft:
```bash
sed -n '1,260p' docs/MVP_Training_Plan.md
sed -n '885,930p' docs/runbook.md
sed -n '1,220p' docs/reviews/Codex-Task-Report_last.md
```
  - Exit-Code: `0`
  - Ergebnis:
    - Stage-4 war bereits in Plan und Runbook erwaehnt
    - fuer Reviewer fehlte aber eine klar benannte eigene Stage-4-Uebersicht
    - der Pflicht-Report war inhaltlich nicht als Reparaturtask formuliert

- Pflicht-Checks nach dem Doku-Fix:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```
  - Exit-Codes: alle `0`
  - Ergebnis:
    - `ruff format`: keine problematischen Aenderungen
    - `ruff check`: alle Checks bestanden
    - `pytest`: Test-Suite erfolgreich

## Erzeugte Artefakte / Zielpfade
- Neue Stage-4-Uebersicht:
  - `docs/BA_MVP_Stage4_Replay_Pipeline.md`
- Bereits vorhandene und jetzt sichtbar verlinkte Stage-4-Artefakte:
  - `configs/coco_replay_ba_mvp_stage4.yaml`
  - `configs/merge_ba_mvp_stage4_with_coco_replay.yaml`
  - `configs/efficientdet_lite2_ba_mvp_stage4.yaml`
  - `work/datasets/coco_replay_ba_v1_stage4/instances_ba_v1.coco.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_combined.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/instances_materialized.json`
  - `work/datasets/ba_mvp_stage4_with_coco_replay/modelmaker.csv`
  - `work/splits/ba_mvp_stage4_with_coco_replay/splits.json`

## Was wurde nur statisch geprüft?
- Es wurde in diesem Reparaturtask bewusst kein neuer Replay-, Merge- oder Trainingslauf gestartet.
- Die bereits vorhandenen Stage-4-Artefakte wurden nur gegen den aktuellen Repo- und Disk-Stand abgeglichen.
- Die naechste Trainingsconfig `configs/efficientdet_lite2_ba_mvp_stage4.yaml` wurde in diesem Task nicht erneut ausgefuehrt.

## Offene Risiken
- Der Branch ist jetzt sichtbarer, aber Stage-4 bleibt weiterhin nur ein vorbereiteter Datenpfad ohne neuen Lite2-Lauf.
- Reviewer muessen fuer die eigentlichen Laufdetails weiter den vorherigen Stage-4-Task-Kontext oder die Artefakte lesen; die neue Uebersicht reduziert das, ersetzt aber keinen Trainingsvergleich.
- `COCO replay` bleibt bewusst schmal; der reale Nutzen muss erst der naechste Stage-4-vs-Stage-3-Trainingsvergleich bestaetigen.

## Nächster sinnvoller Schritt
- Fuehre den echten `EfficientDet-Lite2`-Stage-4-Lauf mit `configs/efficientdet_lite2_ba_mvp_stage4.yaml` aus und dokumentiere den direkten Vergleich gegen die bestehende Stage-3-Baseline.
