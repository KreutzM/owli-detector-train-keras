# BA-v2 Online Augmentation Feasibility

## Scope
- Frage:
  - Kann im aktuellen `EfficientDet-Lite2` / TensorFlow Lite Model Maker Produktpfad eine echte Online-Augmentation fuer den `TRAIN`-Split sauber integriert werden?
- Ziel:
  - keine neue Trainingsarchitektur bauen
  - keine Offline-Duplikation als voreilige Standardantwort setzen
  - den kleinsten belastbaren Hook im bestehenden Pfad identifizieren

## Gepruefter Ist-Pfad
- Repo-Trainingspfad:
  - COCO -> `dataset export modelmaker-csv` -> Model-Maker-CSV
  - `train efficientdet` laedt dieses CSV per `object_detector.DataLoader.from_csv(...)`
  - Model Maker schreibt daraus interne TFRecord-Caches
  - das eigentliche Training laeuft danach ueber EfficientDets internen `InputReader`
- Relevante Repo-Dateien:
  - `src/owli_train/data/modelmaker_csv.py`
  - `src/owli_train/training/modelmaker_efficientdet.py`
  - `src/owli_train/cli.py`
- Lokal gepruefte Model-Maker-Umgebung:
  - `requirements/modelmaker.txt`
  - `tflite-model-maker==0.4.3`
  - `tensorflow==2.8.4`

## Wichtigste technische Beobachtung
- Der aktuelle Repo-Code besitzt keinen eigenen `tf.data`-Trainingsgraphen zwischen CSV-Export und Model-Maker-Training.
- Ein sauberer CPU-seitiger Online-Hook existiert im aktuellen Produktpfad trotzdem:
  - ueber EfficientDet-`hparams` am `model_spec`
  - diese `hparams` werden vom internen EfficientDet-`InputReader` in der Train-Pipeline ausgewertet
- Das ist der kleinste saubere Eingriff, weil:
  - CSV-Export, Label-Order, Eval und Exportpfad unveraendert bleiben
  - keine Fork von Model Maker noetig ist
  - die Box-Transformationen im bereits verwendeten EfficientDet-Dataloader stattfinden

## Bounding Boxes
- Horizontal flip:
  - im aktuellen Pfad sauber moeglich
  - EfficientDets Train-Parser fuehrt `random_horizontal_flip()` auf Bild und Boxen gemeinsam aus
- Random scaling / crop jitter:
  - im aktuellen Pfad sauber moeglich
  - `jitter_min` / `jitter_max` steuern Resize plus zufaellige Crop-/Offset-Wahl
  - die Boxen werden danach ueber `resize_and_crop_boxes()` mit skaliert, verschoben, geclippt und ungueltige Boxen entfernt
- Kleinere Translation:
  - nicht als eigener sauberer Top-Level-Repo-Hook vorhanden
  - praktisch nur indirekt ueber das gleiche Jitter/Crop-Verhalten oder ueber AutoAugment-Policies erreichbar
- Photometrische Augmentierung:
  - keine Box-Anpassung noetig
  - Helligkeit / Kontrast / Farbe sind im aktuellen Produktpfad nicht als einzelne, klare Repo-Schalter verfuegbar
  - sie sind nur indirekt ueber Model-Makers `autoaugment_policy` erreichbar
- AutoAugment:
  - kann sowohl photometrische als auch geometrische Operationen enthalten
  - Boxen werden dort fuer `Translate*_BBox`, `Rotate_BBox`, `Shear*_BBox` mittransformiert
  - das ist technisch vorhanden, aber weniger kontrollierbar als gezielte Einzelknobs

## Was im aktuellen Pfad sauber moeglich ist
- Ein begrenzter Online-Ansatz ist realistisch:
  - `rand_hflip`
  - `jitter_min` / `jitter_max`
  - optional `autoaugment_policy` als experimenteller Schalter
- Das bleibt CPU-seitig in der bestehenden Input-Pipeline.
- Default-Verhalten kann unveraendert bleiben.

## Was im aktuellen Pfad nicht sauber ist
- Eine frei definierbare Repo-eigene Augmentationspipeline mit beliebigen `tf.image`-Ops ist im aktuellen Model-Maker-Pfad nicht klein integrierbar.
- Ein praeziser, separater Brightness-/Contrast-/Color-only-Hook ist aktuell nicht sauber exponiert.
- Ein expliziter, gut kontrollierbarer Translate-only-Schalter existiert im aktuellen Repo-Pfad nicht; dafuer muesste man tiefer in Model-Maker-/EfficientDet-Interna eingreifen oder den Trainingspfad staerker umbauen.

## Kleiner Probe-Hook in diesem Task
- Der Repo-Wrapper akzeptiert jetzt optional:

```yaml
train:
  augmentation:
    rand_hflip: true
    jitter_min: 0.9
    jitter_max: 1.1
    autoaugment_policy: v1
```

- Verhalten:
  - Standardfall ohne `train.augmentation` bleibt unveraendert
  - gesetzte Werte werden nur in `model_spec.hparams` uebersetzt
  - der restliche Produktpfad bleibt gleich
- Der Hook ist bewusst schmal:
  - keine neue Pipeline
  - keine CSV-Umschreibung fuer Augmentation
  - keine Aenderung an Eval/Export

## Risiko-Readout
- Niedriges Integrationsrisiko:
  - fuer `rand_hflip` und moderates `jitter`
  - weil die Logik schon in der verwendeten EfficientDet-Train-Pipeline vorhanden ist
- Mittleres Risiko:
  - fuer `autoaugment_policy`
  - weil die Policy mehrere Operationstypen mischt und dadurch schwerer kontrollierbar ist
- Hohes Refactor-Risiko:
  - fuer alles, was eine eigene Repo-Augmentationspipeline vor Model Maker erzwingen wuerde

## Entscheidung
- Ergebnis:
  - `C) Ein begrenzter Online-Ansatz ist moeglich, aber nur fuer bestimmte Augmentierungstypen`
- Konkrete Lesart:
  - Ja fuer `horizontal flip`
  - Ja fuer moderates `scale/crop jitter`
  - Nur eingeschraenkt und experimentell fuer photometrische Effekte ueber `autoaugment_policy`
  - Nein fuer eine frei gestaltbare, minimal-invasive Repo-eigene Online-Augmentationspipeline im aktuellen Model-Maker-Pfad

## Empfohlene naechste Richtung
- Wenn BA-v2 im bestehenden Produktpfad online augmentiert werden soll:
  - zuerst einen kleinen echten Vergleich mit nur `rand_hflip` plus moderatem `jitter` fahren
  - `autoaugment_policy` nur separat und explizit experimentell pruefen
  - keinen groesseren Trainingsumbau starten, solange dieser begrenzte Hook noch nicht auf kleinem BA-v2-Vergleich validiert ist
