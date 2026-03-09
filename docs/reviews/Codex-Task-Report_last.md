# Codex Task Report

## Ziel
- Den naechsten kleinen, real verifizierten BA-v2-Hazard-Datenbaustein fuer die noch ungestuetzten Hazard-Core-Klassen pruefen und wenn lokal machbar umsetzen.
- Ehrlich klaeren, ob `obstacle_ground` und/oder `obstacle_overhang` auf dem lokalen Repo-Stand belastbar erschliessbar sind.
- Falls nur eine Klasse lokal sauber gestuetzt werden kann, genau diesen kleinen Slice bauen und den Rest explizit offen lassen.

## Was wurde geändert?
- Neue enge Obstacle4-GT-zu-BA-v2-Bootstrap-Map fuer `obstacle_ground` ergaenzt:
  - `configs/label_maps/obstacle4_gt_to_ba_v2_hazard_ground_bootstrap.yaml`
- Neue Slice02-Configs fuer den lokalen Ground-Bootstrap ergaenzt:
  - `configs/balance_ba_v2_hazard_obstacle4_ground_slice02.yaml`
  - `configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground.yaml`
  - `configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground_materialize.yaml`
- Neue dedizierte Slice02-Doku ergaenzt:
  - `docs/BA_v2_Hazard_Slice02_Obstacle4_Ground_Bootstrap.md`
- Bestehende Doku minimal auf Slice02 geschaerft:
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
- Kleine Tests fuer die neue Obstacle4-Ground-Bootstrap-Map und Slice02-Configs ergaenzt:
  - `tests/test_ba_v2_mapping_prep.py`

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - `docs/BA_v2_Hazard_Labelset.md`
  - `docs/BA_v2_Hazard_Slice01_Mapillary_OD.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `configs/label_contracts/ba_v2_hazard.yaml`
  - bestehende BA-v2-Label-Maps und Merge-Configs
  - relevante Datenmodule unter `src/owli_train/data/*`
  - lokale Quellhinweise fuer die fehlenden Klassen:
    - `data/raw/obstacle4/extracted/data.yaml`
    - `data/processed/mapillary_ba_v2_0_sample/instances_ba_v1.coco.json`
    - `work/datasets/od_ba_v1/qc_report.json`
- Inhaltlich verifiziert:
  - lokal ist fuer `obstacle_ground` nur ein enger Bootstrap ueber `Obstacle4` belastbar
  - lokal wurde keine belastbare Quelle fuer `obstacle_overhang` gefunden
  - `Mapillary` und `OD` bleiben fuer `obstacle_overhang` auf aktuellem Repo-Stand unzureichend
  - der neue reale Slice02 stuetzt jetzt:
    - `obstacle_ground`
    - `obstacle_barrier`
    - `obstacle_hole_dropoff`
    - `obstacle_pole`
    - sowie die sechs Rehearsal-Klassen aus Slice01
  - `obstacle_overhang` bleibt auf aktuellem Repo-Stand ungestuetzt
- Real ausgefuehrt:
  - echte BA-v2-Normalisierung des lokalen `Obstacle4`-GT-Exports mit enger Ground-Bootstrap-Map
  - echter BA-v2-Pass-through mit QC fuer den lokalen `Obstacle4`-Ground-Slice
  - echter Merge von `Slice01` plus lokalem `Obstacle4`-Ground-Bootstrap zu einem neuen BA-v2-Slice02
  - echte Nachnormalisierung des kombinierten Slice02 auf kanonische BA-v2-Contract-Reihenfolge
  - echter Split mit `ensure_train_class_coverage`
  - echte COCO-Validierung des kombinierten Slice02 ohne Bilddatei-Existenzpruefung
  - ein realer `materialize_images`-Versuch fuer den vergroesserten Slice02 wurde gestartet, lief lokal aber wiederholt in WSL-Dateisystem-I/O-Waits und wurde deshalb abgebrochen

## Tests
- `PYTHONPATH=src python -c "from owli_train.data.coco import load_coco, load_label_contract_class_names, load_label_map, normalize_coco, write_coco; coco=load_coco('work/datasets/obstacle4/instances_gt.json'); label_map=load_label_map('configs/label_maps/obstacle4_gt_to_ba_v2_hazard_ground_bootstrap.yaml'); order=load_label_contract_class_names('configs/label_contracts/ba_v2_hazard.yaml'); norm=normalize_coco(coco, label_map=label_map, category_order=order); out=write_coco('work/datasets/obstacle4_ba_v2_hazard_ground_source/instances_normalized.json', norm); print(out); print(len(norm['images']), len(norm['annotations']), [c['name'] for c in norm['categories']])"`
  - Exit-Code: `0`
  - Ergebnis: `1250` Bilder, `1627` Annotationen, `4` Kategorien
- `PYTHONPATH=src python -c "from pathlib import Path; from owli_train.data.balance_coco import balance_coco_with_config; a=balance_coco_with_config(Path('configs/balance_ba_v2_hazard_obstacle4_ground_slice02.yaml')); print(a.coco_path); print(a.images, a.annotations, a.categories)"`
  - Exit-Code: `0`
  - Ergebnis: `1250` Bilder, `1627` Annotationen, `4` Kategorien
- `PYTHONPATH=src python -c "from pathlib import Path; from owli_train.data.merge_coco import merge_coco_from_manifest; a=merge_coco_from_manifest(manifest_path=Path('configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground.yaml'), out_path=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json'), report_out_path=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.report.json')); print(a.coco_path); print(a.report_path); print(a.images, a.annotations, a.categories)"`
  - Exit-Code: `0`
  - Ergebnis: `3799` Bilder, `32231` Annotationen, `10` Kategorien
- `PYTHONPATH=src python -c "from owli_train.data.coco import load_coco, load_label_contract_class_names, normalize_coco, write_coco; coco=load_coco('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json'); order=load_label_contract_class_names('configs/label_contracts/ba_v2_hazard.yaml'); norm=normalize_coco(coco, category_order=order); write_coco('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json', norm); print([c['name'] for c in norm['categories']]); print(len(norm['images']), len(norm['annotations']))"`
  - Exit-Code: `0`
  - Ergebnis: kanonische Kategorienreihenfolge mit `obstacle_ground` an Position `1`
- `PYTHONPATH=src python -c "from pathlib import Path; from owli_train.data.coco import load_coco; from owli_train.data.split import split_coco_image_ids, write_splits, write_split_coco_files; coco=load_coco('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json'); splits=split_coco_image_ids(coco, seed=1337, ensure_train_class_coverage=True); out=Path('work/splits/ba_v2_hazard_slice02_mapillary_od_ground'); print(write_splits(out, splits)); write_split_coco_files(out, coco, splits); print({k: len(v) for k, v in splits.items()})"`
  - Exit-Code: `0`
  - Ergebnis: `TRAIN=3039`, `VAL=379`, `TEST=381`
- `PYTHONPATH=src python -c "from owli_train.data.coco import load_coco, validate_coco; s=validate_coco(load_coco('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json')); print(s.images, s.annotations, s.categories, s.category_names)"`
  - Exit-Code: `0`
  - Ergebnis: `images=3799`, `ann=32231`, `cats=10`
- `PYTHONPATH=src python -c "from pathlib import Path; from owli_train.data.materialize_images import materialize_coco_images; a=materialize_coco_images(coco_path=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json'), merge_manifest_path=Path('configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground_materialize.yaml'), out_images_dir=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/images'), out_coco_path=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_materialized.json'), mode='auto'); print(a.out_coco_path); print(a.images_total, a.images_written, a.images_skipped, a.copied, a.symlinked)"`
  - Exit-Code: kein erfolgreicher Exit-Code
  - Ergebnis: lokal gestartet, dann wegen wiederholter WSL-Dateisystem-I/O-Waits manuell abgebrochen
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `67 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: kein erfolgreicher Exit-Code
  - Ergebnis: lokal gestartet, dann im aktuellen WSL-Dateisystemzustand wiederholt in I/O-Wait haengengeblieben und manuell abgebrochen

## Relevante Run-Kommandos
- Lokaler Obstacle4-Ground-Bootstrap:
```bash
PYTHONPATH=src python -c "from owli_train.data.coco import load_coco, load_label_contract_class_names, load_label_map, normalize_coco, write_coco; coco=load_coco('work/datasets/obstacle4/instances_gt.json'); label_map=load_label_map('configs/label_maps/obstacle4_gt_to_ba_v2_hazard_ground_bootstrap.yaml'); order=load_label_contract_class_names('configs/label_contracts/ba_v2_hazard.yaml'); norm=normalize_coco(coco, label_map=label_map, category_order=order); out=write_coco('work/datasets/obstacle4_ba_v2_hazard_ground_source/instances_normalized.json', norm); print(out); print(len(norm['images']), len(norm['annotations']), [c['name'] for c in norm['categories']])"
```
- Slice02-Merge:
```bash
PYTHONPATH=src python -c "from pathlib import Path; from owli_train.data.merge_coco import merge_coco_from_manifest; a=merge_coco_from_manifest(manifest_path=Path('configs/merge_ba_v2_hazard_slice02_mapillary_od_obstacle4_ground.yaml'), out_path=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json'), report_out_path=Path('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.report.json')); print(a.coco_path); print(a.report_path); print(a.images, a.annotations, a.categories)"
```
- Slice02-Split:
```bash
PYTHONPATH=src python -c "from pathlib import Path; from owli_train.data.coco import load_coco; from owli_train.data.split import split_coco_image_ids, write_splits, write_split_coco_files; coco=load_coco('work/datasets/ba_v2_hazard_slice02_mapillary_od_ground/instances_combined.json'); splits=split_coco_image_ids(coco, seed=1337, ensure_train_class_coverage=True); out=Path('work/splits/ba_v2_hazard_slice02_mapillary_od_ground'); print(write_splits(out, splits)); write_split_coco_files(out, coco, splits); print({k: len(v) for k, v in splits.items()})"
```

## Offene Risiken
- `obstacle_overhang` bleibt auf aktuellem lokalen Repo-Stand komplett ungestuetzt.
- `obstacle_ground` ist jetzt real vorhanden, aber nur ueber einen engen Legacy-Bootstrap aus `Obstacle4`.
- Der neue Slice02 ist inhaltlich staerker als Slice01, aber noch immer kein vollstaendiger BA-v2-Hazard-Trainingskandidat fuer den finalen Contract.
- Der vergroesserte Slice02 wurde in diesem Task nicht bis zu einem materialisierten Ein-Root-Bildsatz und `ModelMaker`-CSV durchgezogen, weil der reale Materialisierungsschritt lokal wiederholt in WSL-Dateisystem-I/O-Waits lief.
- Der uebliche Repo-weite `pytest`-Lauf konnte auf diesem lokalen Stand nicht sauber abgeschlossen werden, weil der Prozess im aktuellen WSL-Dateisystemzustand wiederholt in I/O-Wait haengenblieb.
- `Obstacle4` darf durch diesen Schritt nicht wieder zum versteckten Produktanker werden; der Bootstrap muss deshalb eng und explizit begrenzt bleiben.

## Nächster sinnvoller Schritt
- Erschliesse als naechsten kleinen BA-v2-Datenschritt eine belastbare Quelle und enge Mapping-Regel fuer `obstacle_overhang`, bevor ein erster vollstaendiger BA-v2-Hazard-Trainingslauf gestartet wird.
