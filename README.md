# Deep-learning---Hand-Gestrure-recognition

Deep learning engineering Project



Datasettet for prosjektet er for stort vi har derfor lagd klar mapper.



Datasett : https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND\_pub\_v2.zip - inn i trene



Datasett : https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND\_pub\_v2\_eval.zip - inn i annotations



&nbsp;	

Neste milepael i prosjektet er aa verifisere landmarks, ikke aa starte trening med en gang.

Hva koden er satt opp for naa:

- `src/dataset.py` leser FreiHAND RGB-bilder, `training_xyz.json` og `training_K.json`
- den projiserer 3D-handpunkter til 2D-bildekoordinater
- den antar at datasettet har 4 RGB-bilder per annotasjon, siden `training/rgb` har 4x saa mange bilder som landmark-annotasjonene
- `src/inspect_landmarks.py` lagrer et preview-bilde med tegnede handpunkter

Neste praktiske kommando naar Python-miljoeet ditt er tilgjengelig:

```bash
python src/inspect_landmarks.py --index 0
```

Forventet resultat:

- et preview-bilde blir lagret i `outputs/landmark_preview_00000000.jpg`
- punktene skal ligge oppaa handen

Hvis punktene ser riktige ut, er neste steg aa bygge `model.py` og `train.py` for `21 x 2` landmark-prediksjon.

