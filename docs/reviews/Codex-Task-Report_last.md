# Codex Task Report

## Ziel
- Den bisherigen BA-v1-Produktvertrag als historischen verifizierten Interimspfad einordnen, statt ihn weiter als bevorzugte Produktontologie zu behandeln.
- Eine neue hazard-zentrierte BA-v2-Zielontologie fuer das MVP sauber im Repo festlegen.
- Den Repo-Stand so vorbereiten, dass der naechste echte Daten-/Mapping-Schritt nicht mehr an den vier alten Obstacle4-Klassen haengt.

## Was wurde geändert?
- Neuer maschinenlesbarer BA-v2-Hazard-Contract ergaenzt:
  - `configs/label_contracts/ba_v2_hazard.yaml`
  - `configs/label_contracts/ba_v2_hazard.class_names.json`
- Neue Produktdoku fuer die bevorzugte hazard-zentrierte Ontologie ergaenzt:
  - `docs/BA_v2_Hazard_Labelset.md`
- Neue ehrliche Quellen-/Mapping-Strategie fuer BA-v2 hazard ergaenzt:
  - `docs/BA_v2_Hazard_Mapping_Strategy.md`
- Historischen Status von BA-v1 minimal markiert:
  - `docs/BA_v1_Labelset.md`
- Bestehende Produktdoku minimal auf den Ontology-Reset geschaerft:
  - `README.md`
  - `docs/MVP_Training_Plan.md`
  - `docs/runbook.md`
  - `docs/android-export-contract.md`
- Kleinen Konsistenztest fuer BA-v2 hazard ergaenzt:
  - `tests/test_ba_v2_hazard_label_contract.py`
- Ersten konkreten BA-v2-Hazard-Mapping-Slice fuer reale Quellen ergaenzt:
  - `configs/label_maps/mapillary_vistas_to_ba_v2_hazard.yaml`
  - `configs/label_maps/obstacle_dataset_to_ba_v2_hazard.yaml`
  - `configs/label_maps/coco_replay_to_ba_v2_hazard.yaml`
- Kleine BA-v2-Mapping-Prep- und Importer-nahe Tests ergaenzt:
  - `tests/test_ba_v2_mapping_prep.py`
- Verbliebene Reste der verworfenen frueheren Zusatzquelle vollstaendig entfernt:
  - obsolete Label-Map-Datei geloescht
  - historische Hinweise in `docs/BA_v1_Labelset.md`, `docs/MVP_Training_Plan.md` und `docs/runbook.md` entfernt
  - alter Prep-Test in `tests/test_mvp_data_prep.py` entfernt

## Was wurde wirklich verifiziert?
- Statisch geprueft:
  - geforderte BA-v1-/MVP-Dokus
  - bestehende Label-Contracts und Label-Maps
  - relevante Merge-/Trainingsconfigs
  - relevante Daten-, Training-, Eval-, Golden- und TFLite-Label-Resolver im Repo
- Inhaltlich verifiziert:
  - BA-v1 ist im Repo heute vor allem in Doku, Label-Contracts, Label-Maps, Stage-3/4-Configs und Tests fest verankert
  - Eval, Golden und TFLite-Labelauflosung sind positionsbasiert ueber Export-Artefakte und damit grundsaetzlich BA-v2-faehig, ohne dass in diesem Task ein grosser Runtime-Refactor noetig war
  - die neue BA-v2 hazard Ontologie ist im Repo jetzt separat und maschinenlesbar verankert, ohne BA-v1-Historie zu zerstoeren
  - fuer `Mapillary`, `OD` und `COCO replay` existiert jetzt ein erster enger BA-v2-Hazard-Mapping-Slice im Repo
  - die verworfene fruehere Zusatzquelle ist nicht mehr Teil des dokumentierten oder konfigurierten Repo-Pfads
- Real ausgefuehrt:
  - `python -m ruff format .`
  - `python -m ruff check .`
  - `python -m pytest`

## Tests
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `67 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `158 passed, 5 skipped in 4.76s`
- Relevanter neuer Check im Testlauf:
  - `tests/test_ba_v2_hazard_label_contract.py`
  - prueft kanonische Reihenfolge, Rollenpartition, JSON/YAML-Konsistenz und das explizite Ausphasen der alten BA-v1-Obstacle4-Labels
- `tests/test_ba_v2_mapping_prep.py`
  - prueft neue BA-v2-Maps statisch und laesst `Mapillary`, `OD` und `COCO replay` ueber die bestehenden Importpfade mit den neuen Maps laufen
- `tests/test_mvp_data_prep.py`
  - enthaelt keinen alten Prep-Check fuer die verworfene Zusatzquelle mehr

## Relevante Run-Kommandos
- Repo-weite Konsistenzchecks:
```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```

## Offene Risiken
- BA-v2 hazard ist jetzt als Produktziel sauber beschrieben, aber noch nicht datenverifiziert.
- `obstacle_overhang` bleibt der groesste erkennbare Quellluecken-Risiko-Punkt.
- Bestehende Importer und Merge-Pfade arbeiten weiterhin auf BA-v1-benannten Outputs; das ist in diesem Task bewusst nicht global umgebaut worden.
- Die neue Mapping-Strategie ist absichtlich ehrlich und vorlaeufig; sie ersetzt noch keinen real ausgefuehrten Datensatz-Remap.
- Der neue BA-v2-Mapping-Slice deckt bewusst noch nicht `obstacle_ground` oder `obstacle_overhang` ab.
- Historische Dokumente nennen weiter BA-v1-spezifische Klassen und Runs, aber keine verworfene Zusatzquelle mehr als geplanten Pfad.

## Nächster sinnvoller Schritt
- Den naechsten kleinen BA-v2-Quellschritt auf `Obstacle4` als bewusst partielle Bootstrap-Quelle auslegen und dokumentieren, statt seine alten vier Klassen stillschweigend als Endontologie weiterzufuehren.
