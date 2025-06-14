# VisaoComputacionalT2

This repository contains a proof-of-concept pipeline for classifying images from the scikit-learn digits dataset.

The script `digits_mlp.py` demonstrates how to extract HOG features using OpenCV and train an MLP classifier.

## Requirements

- Python 3
- `opencv-python` (provides the `cv2` module)
- `scikit-learn`
- `numpy`

## Usage

Install the dependencies and run the script:

```bash
pip install opencv-python scikit-learn numpy
python3 digits_mlp.py
```

The script will train the model and print a classification report.
