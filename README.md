# Endometriosis Segmentation Project

Deep learning pipeline for segmenting endometriosis lesions in laparoscopic images using U-Net and Swin-UNETR.

## 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 2. Project Structure
* src/: Core logic (config, models, dataset, engine, transforms, visualization).

* scripts/: Executable scripts for training and evaluation.

* notebooks/: Jupyter notebooks for experiments.

* outputs/: Saved models, logs, and visualization results.

## 3. Data Preparation
Setup your .env file with ROBOFLOW_API_KEY. Then, download and process the GLENDA dataset:
```bash
python scripts/prepare_dataset.py
```

## 4. Training
You can train the model using either the CLI script (recommended for performance) or the provided notebook.

### Option A: Command Line (Recommended)
Train with built-in Early Stopping and automatic history logging:
```bash
python scripts/train.py
```

### Option B: Jupyter Notebook
Run the interactive baseline experiment:
```bash
notebooks/01_baseline_experiment.ipynb
```

## 5. Evaluation and Visualization
After training, evaluate the model on the test set and generate visual overlays:

Quantitative Evaluation (Metrics)
```bash
python scripts/evaluate.py
```

Qualitative Visualization (Overlays)
```bash
python scripts/visualize_results.py
```

Check *outputs/test_results/* for side-by-side comparisons (Original, GT, Prediction, Overlay).


## 6. Inference
Perform segmentation on a single image:
```bash
python main.py --image path/to/image.jpg --model unet
```