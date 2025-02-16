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
