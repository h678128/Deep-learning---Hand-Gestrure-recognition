# Deep-learning---Hand-gesture-recognition

Deep learning engineering Project

Dette prosjektet handler om hand landmark- og gesture-gjenkjenning ved hjelp av deep learning. Hovedmodellen er en heatmap-basert CNN, og repoet inneholder både trening, evaluering og demoer lokalt og i nettleser.

Datasettet for prosjektet er for stort vi har derfor lagd klar mapper.



Datasett : https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND\_pub\_v2.zip - inn i trene



Datasett : https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND\_pub\_v2\_eval.zip - inn i annotations


Modellen kan bli brukt her: https://h678128.github.io/Deep-learning---Hand-gesture--recognition/

Men det anbefales å laste ned desktop versjonen for en bedre opplevelse, ettersom denne gir en raskere response.


- Python-versjon: prosjektet er testet med Python 3.12
- `torch`: brukes til å laste og kjøre modellen
- `opencv-python`: brukes til bildebehandling og live kamera
- `mediapipe`: brukes som hand-prefilter i noen demoer
- `flask`: brukes i den lokale web-appen
- `numpy`: brukes til numeriske operasjoner
- `matplotlib`: brukes til visualisering og inspeksjon under utvikling
- `pyautogui`: brukes til musekontroll i desktop-demoen
- avhengighetene installeres via `requirements.txt`

```text
Deep-learning---Hand-gesture-recognition/
├── data/                  - datasettmapper for FreiHAND og eventuelt annet datagrunnlag
│   ├── trene/             - FreiHAND train-data
│   ├── annotations/       - FreiHAND eval / annotasjoner
│   ├── images/            - Ultralytics-bilder brukt i noen eksperimenter
│   ├── labels/            - Ultralytics-labels brukt i noen eksperimenter
│   └── data.yaml          - datasettbeskrivelse for Ultralytics-data
├── docs/                  - GitHub Pages-versjonen av prosjektet
│   ├── index.html         - nettleserdemoen
│   └── model.onnx         - ONNX-modellen brukt av Pages-demoen
├── modell/                - lagrede modellfiler og checkpoints lokalt
├── outputs/               - genererte evalueringer, prediksjoner og andre outputs
├── src/                   - all hovedkode for trening, evaluering og demoer
│   ├── dataset.py         - datasettlasting, heatmaps, cropping og augmentering
│   ├── model.py           - modellarkitekturene
│   ├── train.py           - trening og lagring av checkpoints
│   ├── evaluate.py        - evaluering av modell på FreiHAND
│   ├── predict_folder.py  - prediksjon på en mappe med bilder
│   ├── webcam_live.py     - lokal live-demo med kamera
│   ├── web_app.py         - lokal Flask web-app
│   ├── gesture.py         - gesture-logikk basert paa landmarks
│   ├── export_onnx.py     - eksport fra PyTorch til ONNX
│   ├── inspect_landmarks.py - visuell inspeksjon av landmarks
│   ├── compare_mappings.py - sammenligner landmark-mappinger
│   └── test_dataset.py    - enkel testing av datasettoppsett
├── .gitignore             - ignorerer store datafiler, modeller og outputs
├── requirements.txt       - Python-avhengigheter
├── start.bat              - enkel oppstart av desktop-demo paa Windows
└── README.md              - prosjektbeskrivelse
```

Modellen vi bruker til å vise prosjektet er `modell\landmark_heatmap11_best.pt`. Den beste modellen kom på epoke 38 med `3.97` validation pixel error.

En rolig og enkel bakgrunn fungerer ofte bedre enn en bakgrunn med mye variasjon. Mye stoy, mange objekter eller store variasjoner i bildet kan gjøre modellen mindre stabil. Ansiktet kan også noen ganger lage utfordringer, spesielt hvis det er nært hånden.

Nyttige kommandoer:

```powershell
# live-demo
python src\webcam_live.py --checkpoint modell\landmark_heatmap11_best.pt
python src\webcam_live.py --checkpoint modell\landmark_heatmap11_best.pt --use-mediapipe

# lokal web-app
python src\web_app.py --default-checkpoint modell\landmark_heatmap11_best.pt

# evaluer modellen
python src\evaluate.py --checkpoint modell\landmark_heatmap11_best.pt --index 25
```

Demovideo: https://youtube.com/shorts/qS3M5z2wxhs

Bilder fra prosjektet. Fra tildig fase til slutt: 

<img width="520" alt="image" src="https://github.com/user-attachments/assets/12cf2a2c-4265-4469-a7a9-b2b1fa8981c0" />

<img width="520" alt="image" src="https://github.com/user-attachments/assets/2cedc295-f356-404a-8945-eef043c0a048" />

<img width="520" alt="image" src="https://github.com/user-attachments/assets/5e98b576-78d5-408b-95a5-9944f22fc6dc" />

<img width="520" alt="image" src="https://github.com/user-attachments/assets/cda08634-6ce2-4d09-8bc1-a64edb59b019" />

<img width="520" alt="image" src="https://github.com/user-attachments/assets/45369909-9d7e-4d67-b051-6118d3351a97" />



