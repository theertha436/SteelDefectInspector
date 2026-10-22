# Change log — stretch-goal update

- Added `NEU-DET Bounding Boxes` page to Streamlit navigation.
- Added Pascal VOC XML parser for NEU-DET annotation files and image overlay renderer.
- Viewer can browse a local NEU-DET image folder and match same-stem XML files, or accept an uploaded image plus its matching XML annotation.
- Clearly labels overlays as ground-truth annotations; this update does not claim model-generated object detection.
- Improved shift log visualization with a shift-by-predicted-class bar chart; full records and CSV export remain available.
- Class Management already supports rejecting/deactivating an added class and merging its prototype into a target class, with action history.
- Added unit test for XML parsing and box rendering.

## Validation

`python -m compileall -q app.py neu_det.py steel_core.py train_model.py`

`python -m pytest -q tests` → 6 passed in the authoring environment.

## Dataset requirement

Download NEU-DET separately and keep its image files and Pascal VOC XML annotation files together in a consistent folder layout. Point the UI at that folder. Ground-truth visualization is implemented; a trained detector's prediction boxes require a separate detector-training/inference implementation and are not claimed here.
