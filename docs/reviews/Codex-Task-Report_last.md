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
- Real ausgefuehrt:
  - `python -m ruff format .`
  - `python -m ruff check .`
  - `python -m pytest`

## Tests
- `python -m ruff format .`
  - Exit-Code: `0`
  - Ergebnis: `66 files left unchanged`
- `python -m ruff check .`
  - Exit-Code: `0`
  - Ergebnis: `All checks passed!`
- `python -m pytest`
  - Exit-Code: `0`
  - Ergebnis: `153 passed, 5 skipped in 4.67s`
- Relevanter neuer Check im Testlauf:
  - `tests/test_ba_v2_hazard_label_contract.py`
  - prueft kanonische Reihenfolge, Rollenpartition, JSON/YAML-Konsistenz und das explizite Ausphasen der alten BA-v1-Obstacle4-Labels

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

## Nächster sinnvoller Schritt
- Einen kleinen, quellenbezogenen BA-v2-Hazard-Mapping-Slice umsetzen und lokal verifizieren, beginnend mit den saubersten Kandidaten fuer `obstacle_barrier`, `obstacle_pole` und den sechs Rehearsal-Klassen, bevor ein neuer Trainingslauf gestartet wird.
