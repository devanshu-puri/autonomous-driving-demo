# Automobile Driving Decision Demo

This project is a complete Python demo for an automobile automation topic in image processing. It uses a real subset of the open-source `comma10k` autonomous-driving dataset, learns a lightweight scene-understanding model from the dataset masks, and then predicts what the car should do while driving.

The project is designed for:

- academic demos
- mini-project presentations
- explainable AI demonstrations
- learning how image understanding can be connected to driving decisions

## What The Project Does

Given a road image, the pipeline:

1. predicts a coarse semantic scene map from the image
2. measures driving-relevant conditions such as:
   - drivable area in front
   - lane visibility
   - movable objects ahead
   - road obstruction on the left and right
3. predicts a driving action:
   - `KEEP_STRAIGHT`
   - `SLOW_DOWN`
   - `STOP`
   - `MOVE_LEFT`
   - `MOVE_RIGHT`
   - `CAUTION`
4. explains why that decision was made

## Dataset

This project includes a packaged subset of the open-source `comma10k` dataset:

- source: [commaai/comma10k](https://github.com/commaai/comma10k)
- license: MIT
- included here: 60 real driving images and their matching segmentation masks

The dataset subset is stored in:

- `data/comma10k_subset/images`
- `data/comma10k_subset/masks`

## Project Structure

```text
automobile_decision_ai/
├─ app.py
├─ demo.py
├─ train.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ comma10k_subset/
├─ models/
├─ outputs/
├─ src/
│  └─ automobile_decision_ai/
└─ tests/
```

## How The AI Decides

This demo uses a two-stage pipeline:

### 1. Scene Understanding Model

A lightweight Random Forest model is trained from the dataset images and segmentation masks. It learns to classify pixels into these road-scene classes:

- road
- lane marking
- undrivable area
- movable object
- ego vehicle

### 2. Driving Policy Model

From the predicted scene map, the system extracts traffic-aware features and predicts a driving action with a Decision Tree policy model.

This makes the demo explainable: you can see the image, the predicted scene understanding, and the final action.

## Run The Project

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
python train.py
python demo.py
python app.py
```

## Commands

### Train everything

```bash
python train.py
```

This will:

- create metadata
- train the segmentation model
- train the driving policy model
- save metrics to `outputs/metrics.json`
- save per-sample predictions to `outputs/predictions.csv`

### Run one sample demo

```bash
python demo.py
```

Optional:

```bash
python demo.py --sample 1312_99c94dc769b5d96e_2018-07-25--20-08-21_7_445
```

### Launch the web demo

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Outputs

After training you will get:

- `models/segmentation_random_forest.joblib`
- `models/policy_decision_tree.joblib`
- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/demo_*.png`

## Important Note

This is an educational prototype for project demonstration purposes. It is not a real autonomous-driving system and must never be used to control a real vehicle.
