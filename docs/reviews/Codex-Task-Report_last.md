# Codex Task Report

## Ziel
- Die neue Produktentscheidung sauber im Repo durchziehen, dass `obstacle_overhang` vorerst nicht Teil des BA-v2 MVP ist.
- Den BA-v2-Produktvertrag auf vier Hazard-Core-Klassen plus sechs Rehearsal-Klassen festziehen.
- Den vorhandenen BA-v2-Datenstand bis direkt vor den ersten Lite2-Trainingslauf als echten BA-v2-MVP-Trainingskandidaten vorbereiten.

## Was wurde geändert?
- BA-v2-MVP-Contract ohne `obstacle_overhang` aktualisiert:
  - `configs/label_contracts/ba_v2_hazard.yaml`
  - `configs/label_contracts/ba_v2_hazard.class_names.json`
- BA-v2-MVP-Doku geschaerft:
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/BA_v2_Hazard_Mapping_Strategy.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/android-export-contract.md`
- Historische Slice-Dokus minimal nachgeschaerft, damit klar bleibt, dass sie unter dem frueheren fuenf-Kern-Klassen-Stand entstanden:
  - `docs/BA_v2_Hazard_Slice01_Mapillary_OD.md`
  - `docs/BA_v2_Hazard_Slice02_Obstacle4_Ground_Bootstrap.md`
- Neue Doku fuer den ersten train-ready BA-v2-MVP-Kandidaten ergaenzt:
  - `docs/BA_v2_MVP_Train_Candidate.md`
- Neue Trainingsconfig fuer den naechsten Lite2-Lauf ergaenzt:
  - `configs/efficientdet_lite2_ba_v2_mvp.yaml`
- BA-v2-bezogene Label-Map-Hinweise auf den neuen MVP-Contract nachgezogen:
  - `configs/label_maps/mapillary_vistas_to_ba_v2_hazard.yaml`
  - `configs/label_maps/obstacle_dataset_to_ba_v2_hazard.yaml`
  - `configs/label_maps/obstacle4_gt_to_ba_v2_hazard_ground_bootstrap.yaml`
  - `configs/balance_ba_v2_hazard_obstacle4_ground_slice02.yaml`
- Kleine statische Tests fuer den neuen Contract und den Trainingsconfig-Pfad ergaenzt bzw. angepasst:
  - `tests/test_ba_v2_hazard_label_contract.py`
  - `tests/test_ba_v2_mapping_prep.py`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/BA_v2_Hazard_Slice01_Mapillary_OD.md`
  - `docs/BA_v2_Hazard_Slice02_Obstacle4_Ground_Bootstrap.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/android-export-contract.md`
  - `configs/label_contracts/ba_v2_hazard.yaml`
  - BA-v2-relevante Label-Maps und Merge-Configs unter `configs/label_maps/*` und `configs/*`
  - Trainings-/Eval-/Golden-/TFLite-Pfade unter:
    - `src/owli_train/training/*`
    - `src/owli_train/eval/*`
    - `src/owli_train/golden/*`
    - `src/owli_train/tflite_detect.py`
- Inhaltlich verifiziert:
  - `obstacle_overhang` war auf aktuellem Repo-Stand nur noch Teil des bevorzugten BA-v2-Contracts und der dazugehoerigen Doku/Tests, nicht eines real gestuetzten Datenpfads
  - Trainings-/Eval-/Golden-/TFLite-Pfade verwenden artefaktbasierte Klassenlisten und benoetigten keinen BA-v2-spezifischen Codeumbau fuer das Entfernen von `obstacle_overhang`
  - der aktuelle BA-v2-Datenstand nach dem Ground-Bootstrap stuetzt real:
    - `obstacle_ground`
    - `obstacle_barrier`
    - `obstacle_hole_dropoff`
    - `obstacle_pole`
    - sowie die sechs Rehearsal-Klassen
  - unter dem neuen vier-Klassen-MVP-Contract ist dieser Datenstand jetzt der erste echte BA-v2-MVP-Trainingskandidat
- Real ausgefuehrt:
  - Materialisierung des BA-v2-MVP-Kandidaten aus dem bestehenden Slice02-COCO
  - Validierung des materialisierten BA-v2-MVP-COCO mit Bilddatei-Pruefung
  - Export des materialisierten BA-v2-MVP-Kandidaten zu ModelMaker-CSV
  - Repo-weite Format-/Lint-/Test-Verifikation

## Tests
- `PYTHONPATH=src python -m owli_train dataset materialize-images --coco work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json --merge-manifest configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground_materialize.yaml --out-images-dir work/datasets/ba_v2_mvp_candidate/images --out-coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json --mode symlink`
  - Exit-Code: `0`
  - Ergebnis: `3799` Bilder materialisiert, `3799` Symlinks geschrieben
- `PYTHONPATH=src python -m owli_train dataset validate --coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json --images-dir work/datasets/ba_v2_mvp_candidate/images`
  - Exit-Code: `0`
  - Ergebnis: `images=3799`, `ann=32231`, `cats=10`
- `PYTHONPATH=src python -m owli_train dataset export modelmaker-csv --coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json --images-dir work/datasets/ba_v2_mvp_candidate/images --splits-json work/splits/ba_v2_hazard_slice02_mapillary_od_ground/splits.json --out work/datasets/ba_v2_mvp_candidate/modelmaker.csv`
  - Exit-Code: `0`
  - Ergebnis: `rows=32231`, `images=3799`, `annotations=32231`
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `67 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `165 passed, 5 skipped in 5.14s`

## Relevante Run-Kommandos
- Materialize BA-v2-MVP candidate:
```bash
PYTHONPATH=src python -m owli_train dataset materialize-images \
  --coco work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json \
  --merge-manifest configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground_materialize.yaml \
  --out-images-dir work/datasets/ba_v2_mvp_candidate/images \
  --out-coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json \
  --mode symlink
```
- Validate BA-v2-MVP candidate:
```bash
PYTHONPATH=src python -m owli_train dataset validate \
  --coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images
```
- Export ModelMaker CSV:
```bash
PYTHONPATH=src python -m owli_train dataset export modelmaker-csv \
  --coco work/datasets/ba_v2_mvp_candidate/instances_materialized.json \
  --images-dir work/datasets/ba_v2_mvp_candidate/images \
  --splits-json work/splits/ba_v2_hazard_slice02_mapillary_od_ground/splits.json \
  --out work/datasets/ba_v2_mvp_candidate/modelmaker.csv
```

## Offene Risiken
- `obstacle_ground` ist zwar jetzt Teil des train-ready MVP-Kandidaten, bleibt aber datenmaessig ein enger Legacy-Bootstrap ueber `Obstacle4`.
- `obstacle_barrier` und `obstacle_hole_dropoff` bleiben im Vergleich zu `obstacle_pole` und den Rehearsal-Klassen kleiner und ungleich verteilt.
- `obstacle_overhang` ist nicht geloest, sondern bewusst aus dem MVP entfernt; eine spaetere Rueckkehr waere eine neue Produkt- und Datenentscheidung.
- Es wurde in diesem Task bewusst kein Lite2-Trainingslauf gestartet, daher existiert noch kein verifizierter BA-v2-MVP-Trainingsreport oder Export-Artefakt.

## Nächster sinnvoller Schritt
- Starte den ersten echten BA-v2-MVP-Lite2-Trainingslauf mit `configs/efficientdet_lite2_ba_v2_mvp.yaml` auf dem materialisierten Kandidaten unter `work/datasets/ba_v2_mvp_candidate`.
