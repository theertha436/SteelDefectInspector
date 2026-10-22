# SteelDefectInspector — implementation and operator guide

## Problem statement coverage

- **Initial training:** five classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, and `rolled_in_scale`.
- **Held-out class:** `scratches`. It is excluded from initial training and is treated as unknown before a prototype is added.
- **Unknown detection:** cosine similarity to training-derived class centroids, with the acceptance threshold calibrated using known validation images and `unknown_validation` scratches only. Test data and five-shot examples are not used for threshold calibration.
- **Operator UI:** upload an image, view the predicted class or `unknown`, review similarity scores, and optionally save an inspection record.
- **Few-shot learning:** operator can upload exactly five labelled images or use the reserved five-image Scratches pool. The system creates a normalized prototype and a class-specific threshold from support embeddings without retraining the neural network.
- **Class management:** added classes can be rejected/deactivated or merged into another active class. Class changes are recorded in `saved_models/class_bank.json` history.
- **Shift log:** inspections can be saved with a Day/Evening/Night shift; the Inspection Log page shows counts by shift and predicted defect, a chart, full records, and CSV export.
- **Evaluation UI:** readable metric cards, a Precision–Recall–F1 table, confusion matrix and training history; raw JSON is optional.

## Dataset split

The `prepare_dataset.py` script expects the NEU-CLS classification set: 1,800 grayscale images, 300 per class. It creates:

| Split | Count | Use |
|---|---:|---|
| `train` | 1,050 | 210 images per known class; scratches excluded |
| `validation` | 225 | Known-class validation and threshold calibration |
| `test` | 225 | Known-class test set |
| `unknown_validation` | 45 | Threshold calibration only |
| `few_shot_pool` | 5 | Operator support examples, excluded from test |
| `unknown_test` | 250 | Held-out scratches test before update; used as new-class test after update |

NEU-CLS images are grayscale. `load_image()` converts to RGB, effectively repeating the grayscale channel into three channels for ImageNet-pretrained ResNet-18. The dataset prep script expects class names in filenames; inspect and adjust `get_class()` if your download names images differently.

## Windows CMD setup

From the project root:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python run_all.py --quick
python -m streamlit run app.py
```

After the quick smoke test passes, run the normal training pipeline for the configured eight epochs:

```bat
python run_all.py
```

The first ImageNet weight download may need internet access. If unavailable, training warns and falls back to random initialization; check `outputs/metrics.json` before reporting results. The quick run is a pipeline test, not a final experiment.

## Operator workflow

1. Open **Inspect Image** and upload a steel surface image.
2. Review the class prediction or `Unknown defect — operator review recommended`.
3. To add a class, open **Few-Shot Learning**, upload exactly five images with the same operator-confirmed label, enter the class name, confirm the label, and create/update the prototype. The reserved Scratches pool is also available for the benchmark class.
4. The new class is saved with a class-specific threshold estimated from the five support embeddings. This is a heuristic, not a calibrated probability; review the results on held-out test images.
5. Open **Model Evaluation** to inspect original known-vs-unknown metrics and the confusion matrix.
6. Run the post-update evaluation:

```bat
python evaluate_after_few_shot.py
```

It writes `outputs/few_shot_evaluation.json`. It tests known test images and the 250 held-out scratches test images; it does not use the five support images. Once scratches has been learned, this post-update report measures scratches recognition, not rejection of a *different* unknown defect class.
7. Save inspections from the Inspect Image page to build `outputs/inspection_log.csv`; the Inspection Log page shows counts by shift and supports CSV export.
8. In **Class Management**, deactivate a class or merge its prototype into another class. These actions are recorded in class-bank history.

## Important limitations / optional stretch goal

- The required MVP is classification plus open-set unknown detection and few-shot prototype addition. It is not a certified industrial quality-control system.
- **Detection model limitation:** the app visualizes NEU-DET ground-truth boxes from the actual XML annotations. It does not yet train or run a learned object detector, so describe this as annotation/bounding-box visualization rather than AI-generated detection.
- Similarity scores are not calibrated probabilities. The class-specific threshold for a newly added class is estimated from five support examples and should be validated on representative labelled data.
- The current five-shot workflow supports one prototype per label. Merge behavior combines prototypes; it is a simple operator utility rather than a statistically validated incremental-learning method.
- Dataset reference requested by the problem statement: Song and Yan (2013), NEU Surface Defect Database. Add the complete bibliographic entry supplied by the hackathon organizer/dataset page to the final submission references, and verify the exact citation details before submission.
