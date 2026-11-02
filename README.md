# Wound Segmentation

Wound segmentation capstone, three parts in one repository.

- `app/`: browser app that measures a wound from a photo and tracks healing over time. A QR code of known size gives the scale. Dummy data only. Not medical advice. Run it: `cd app && python3 -m http.server 8000`, then open http://localhost:8000.
- `model/`: training and evaluation pipeline for YOLOv8n-seg wound segmentation. Package `src/`, tests, Kaggle notebooks for the baseline and ablations 1 to 6, experiment notes, result reports. Run it: `cd model && pip install -r requirements.txt && python -m pytest`.
- `benchmark/`: three benchmark notebooks (domain adaptation, factor analysis, TinyML web deployment). Run it: `cd benchmark && pip install -r requirements.txt && python validate_notebook.py act-1-clean.ipynb`.

Each part has its own README with the exact commands and test counts under "Run it".

Other notes:
- `docs/DATA.md`: where the images and datasets come from.
- `docs/OVERLAP.md`: what repeated across the three parts, what was removed, what is unique.

Status: work in progress.

## AI use disclosure

I used an AI coding assistant to prepare this repository for presentation: dependency updates, security fixes, rewriting documentation and comments, and running the tests. The design and the original code are mine. The history here starts at the cleaned version.
