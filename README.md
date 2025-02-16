# VLM_case_study
This repo contains solutions for the sensmore interview case study

## Installation

### Prerequisites

Note that this repo is only tested on mac.

Ensure you have **Python 3.10** installed. If not, you can install it using:

```bash
sudo apt update && sudo apt install python3.10 python3.10-venv python3.10-dev  # Ubuntu/Debian
brew install python@3.10  # macOS
```

### Install `ffmpeg`

This project requires `ffmpeg` for media processing. Install it using:

```bash
brew install ffmpeg  # macOS
sudo apt install ffmpeg  # Ubuntu/Debian
```

### Setting Up the Virtual Environment

It's recommended to create a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows (PowerShell)
```

### Install Dependencies

Run the following command to install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Now you're ready to use the project! 🚀

# VLM Training Pipeline

## Overview
This script orchestrates a pipeline for generating, processing, and training a Visual Language Model (VLM). It consists of five main steps: dataset generation, VQA pair generation, YOLO bounding box creation, VLM training, and VLM testing.

## Steps
1. **Dataset Generation** (Optional)
   - Downloads and processes video data.
   - Functions: `download_all_videos()`, `process_videos()`.

2. **VQA Pair Generation** (Optional)
   - Generates Visual Question Answering (VQA) pairs using a VLM.
   - Function: `generate_vqa()`.

3. **YOLO Bounding Box and Action Generation** (Optional)
   - Processes images to create YOLO bounding boxes and action labels.
   - Function: `process_images()`.

4. **Train the VLM** (Required)
   - Trains the Visual Language Model.
   - Function: `train_vlm.train_model()`.

5. **Test the VLM** (Optional)
   - Tests the trained VLM model.
   - Function: `test_vlm.test_model()`.

## Usage
Modify the boolean flags in the script to enable/disable specific steps:

```python
GEN_DATA = False  # Set to True to generate dataset
GEN_VQA_PAIRS_USING_VLM = False  # Set to True to generate VQA pairs
GEN_YOLO_BBOXES_AND_ACTION = False  # Set to True to generate YOLO bounding boxes
TRAIN_VLM = True  # Set to True to train the VLM
TEST_VLM = False  # Set to True to test the VLM
```

Run the script:
```bash
python main.py
```

Ensure all dependencies are installed before running the script.

## Notes
- The script will only execute the steps where the corresponding flags are set to `True`.
- Training the VLM is a required step in this pipeline.
- Modify the script as needed to suit your data and model requirements.

