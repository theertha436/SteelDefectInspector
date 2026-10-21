
# SteelDefectInspector

AI-powered steel surface inspection for known-defect classification,
unknown-defect detection, and five-shot learning.

## Problem
Traditional defect classifiers may force an unfamiliar surface defect
into one of their known categories.

## Proposed solution
- Classify five known steel surface defect categories.
- Flag samples that do not sufficiently match known classes.
- Allow an operator to label five examples of a new defect class.
- Update recognition using a stored feature representation without
  retraining the entire neural network.

## Dataset
NEU-CLS steel surface defect dataset.

The project currently uses a held-out class for experimentation.
The dataset preparation and evaluation results will be documented
as they are verified.

## Technology stack
- Python
- PyTorch and Torchvision
- Streamlit
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn

## Project status
- [x] Python virtual environment created
- [x] Core libraries installed
- [x] Dataset extracted and inspected
- [ ] Dataset preparation verified
- [ ] Classifier trained and evaluated
- [ ] Unknown-defect detection implemented
- [ ] Five-shot learning implemented
- [ ] Streamlit interface completed
- [ ] Free deployment completed

## Run locally
Setup and execution instructions will be updated as the application
is implemented and tested.
