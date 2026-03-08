# Codex Task Report

## Ziel
- Das Repo auf einen klaren primaeren BA-MVP-Trainingspfad ausrichten, damit nach Abschluss der laufenden Downloads die neuen Datenquellen ohne weiteren Grundsatzumbau integriert werden koennen.
- Den naechsten Multi-Source-Pfad explizit machen: `Obstacle4` als verifizierte Baseline, dazu `Mapillary Vistas`, `TACO`, `Obstacle-Dataset / OD` und ein kleines `COCO replay` fuer die sechs BA-v1-Rehearsal-Klassen.

## Was wurde geändert?
- Neue Planungsdoku angelegt: `docs/MVP_Training_Plan.md`
- Neue konservative Prep-Configs angelegt:
  - `configs/label_maps/mapillary_vistas_to_ba.yaml`
  - `configs/label_maps/taco_to_ba.yaml`
  - `configs/label_maps/coco_replay_to_ba.yaml`
- Kleinen Konsistenztest fuer die neuen Prep-Dateien ergaenzt: `tests/test_mvp_data_prep.py`
- Bestehende Doku minimal auf den neuen Primary Path geschaerft:
  - `README.md`
  - `docs/BA_v1_Labelset.md`
  - `docs/runbook.md`
- `docs/reviews/Codex-Task-Report_last.md` auf diesen Taskstand aktualisiert.

## Was wurde wirklich verifiziert?
- Tatsaechlich ausgefuehrte Kommandos:
  - `sed -n '1,220p' README.md`
  - `sed -n '1,260p' docs/runbook.md`
  - `sed -n '1,240p' docs/BA_v1_Labelset.md`
  - `sed -n '1,260p' docs/Obstacle4_E2E_Results.md`
  - `sed -n '1,220p' docs/android-export-contract.md`
  - `sed -n '1,240p' configs/label_contracts/ba_v1.yaml`
  - `ls -1 configs/label_maps && ...`
  - `sed -n '1,220p' configs/merge_obstacle4_gt_pseudo.yaml`
  - `sed -n '1,260p' configs/efficientdet_lite2_obstacle4.yaml`
  - `ls -1 src/owli_train/data && ...`
  - `ls -1 tests && ...`
  - `sed -n '1,240p' docs/Obstacle_Dataset_Integration.md`
  - `sed -n '1,260p' docs/review-templates/Codex-Task-Report.md`
  - `find data -maxdepth 3 -type d | sort | sed -n '1,300p'`
  - `find work -maxdepth 3 -type d | sort | rg 'obstacle4|mapillary|taco|obstacle|coco' -n -S`
- Ergebnis der realen lokalen Pruefung:
  - `Obstacle4` ist der einzige bereits voll materialisierte BA-Produktpfad in `data/raw`.
  - `data/coco2017` ist lokal vorhanden und kann spaeter als Replay-Quelle dienen.
  - `Mapillary Vistas`, `TACO` und `OD` sind fuer diesen Task nicht als belastbar integrierte Rohdatenquelle im Repo materialisiert; der Patch bleibt deshalb bewusst bei Doku-/Config-Prep.
- Rein statisch geprueft:
  - Die neuen `label_maps` bleiben auf den BA-v1-Contract begrenzt.
  - `COCO replay` ist explizit als schmaler Rehearsal-Baustein und nicht als COCO-80-Rueckfall festgelegt.
  - Die bestehende Obstacle4-Baseline bleibt unveraendert der einzige voll verifizierte Trainingspfad.

## Tests
- `python -m ruff format .`
  - Exit-Code: `0`
  - Resultat: `1 file reformatted, 58 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Resultat: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Resultat: `123 passed, 5 skipped`

## Relevante Run-Kommandos
- Kein neuer Multi-Source-Lauf wurde in diesem Task real ausgefuehrt, weil die zusaetzlichen Rohdaten-Downloads noch nicht als verifizierte lokale Eingaben in den Repo-Pfad uebernommen wurden.
- Vorbereitete naechste Kommandos nach Abschluss der lokalen Source-Reviews:
```bash
python -m owli_train dataset normalize \
  --coco <verified_mapillary_or_taco_or_od_coco.json> \
  --images-dir <verified_images_dir> \
  --label-map configs/label_maps/<source>_to_ba.yaml \
  --out work/datasets/<source>/instances_ba_v1.json

python -m owli_train dataset validate \
  --coco work/datasets/<source>/instances_ba_v1.json \
  --images-dir <verified_images_dir>
```
- Fuer das spaetere Rehearsal-Subset ist vorbereitet:
  - `configs/label_maps/coco_replay_to_ba.yaml`
  - Zielklassen: `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`

## Offene Risiken
- Die tatsaechlichen lokalen Rohpfade, Formate und Taxonomien fuer `Mapillary Vistas`, `TACO` und `OD` sind in diesem Task noch nicht verifiziert.
- Ohne diese realen Source-Reviews bleiben die neuen Mapping-Dateien bewusst konservative Stubs.
- Das konkrete COCO-Replay-Subset ist noch nicht materialisiert; nur sein enger Klassenvertrag ist jetzt explizit vorbereitet.
- Der aktuell einzige voll verifizierte Trainingslauf bleibt weiterhin `Obstacle4` plus Pseudo-Labels.

## Nächster sinnvoller Schritt
- Nach Abschluss der laufenden Downloads zuerst einen der neuen Datensaetze lokal verifizieren, dessen reale Klassenliste gegen die passende `configs/label_maps/*.yaml` mappen und einen ersten normalisierten `instances_ba_v1.json`-Zwischenstand in den MVP-Pfad einhaengen.
