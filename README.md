# YOLO-Pipeline: Automated Object Detection Training Framework

An end-to-end pipeline for automating YOLO object detection model training workflows, from data acquisition to model training and evaluation.

## Features

- **Data Acquisition**: Download datasets from Google Drive with automatic extraction and validation
- **Data Validation**: Verify dataset structure and create YAML configuration files
- **Model Training**: Train various YOLO versions (v5-v11) with GPU acceleration
- **Model Size Selection**: Choose from nano, small, medium, large, or xlarge model variants
- **GPU Optimization**: Automatic GPU detection and memory optimization
- **Real-time Monitoring**: Track GPU usage during training

## Installation

```bash
# Clone the repository
git clone https://github.com/Nandha-06/YOLO-Pipeline-Automated-Object-Detection-Training.git
cd YOLO-Pipeline-Automated-Object-Detection-Training

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

Run the complete pipeline with a single command:

```bash
python main.py
```

This will guide you through:
1. Downloading and extracting the dataset
2. Validating the dataset structure
3. Training a YOLO model with your preferred settings

## Usage Guide

### Data Ingestion

Download and prepare datasets from Google Drive.

#### Interactive Mode
```bash
python data_ingestion.py
```
The script will prompt you for a Google Drive URL and automatically validate the downloaded dataset.

#### Command Line Mode
```bash
# Download a file
python data_ingestion.py "https://drive.google.com/file/d/FILEID/view"

# Download a file with custom output path
python data_ingestion.py "https://drive.google.com/file/d/FILEID/view" -o "data/file.zip"

# Download a folder
python data_ingestion.py "https://drive.google.com/drive/folders/FOLDERID" -f

# Download multiple files
python data_ingestion.py "URL1" "URL2" "URL3"

# Download without extracting zip files
python data_ingestion.py "URL" -n

# Download and create YAML if missing
python data_ingestion.py "URL" -y

# Keep zip files after extraction
python data_ingestion.py "URL" -k
```

#### Options
- `-q` Quiet mode
- `-o` Output path
- `-f` Folder mode
- `-n` No extraction (don't extract zip files)
- `-v` Validate dataset after downloading
- `-y` Create YAML template if missing (disabled by default)
- `-k` Keep zip files after extraction (by default, zip files are deleted)

### Data Validation

Validate if your dataset has the necessary files and structure for YOLO training.

```bash
# Validate using last extracted directory
python data_validation.py

# Validate specific directory
python data_validation.py "path/to/dataset"

# Create a YAML template if missing
python data_validation.py --create-yaml
```

### Model Training

Train YOLO models with GPU acceleration and customizable settings.

```bash
# Interactive training with prompts
python model_training.py

# Specify model size and batch size
python model_training.py --size m --batch 8

# Specify dataset directory and number of epochs
python model_training.py --data_dir "path/to/dataset" --epochs 100

# Train with specific image size
python model_training.py --img_size 640

# Force GPU usage and monitor performance
python model_training.py --force-gpu --monitor-gpu
```

#### Options
- `--size` Model size (n, s, m, l, x)
- `--batch` Batch size
- `--epochs` Number of training epochs
- `--img_size` Image size for training
- `--data_dir` Dataset directory
- `--weights` Pre-trained weights file
- `--device` Device to train on (0 for GPU, cpu for CPU)
- `--verbose` Enable verbose output
- `--force-gpu` Force GPU usage even if checks fail
- `--monitor-gpu` Monitor GPU usage during training

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- NVIDIA drivers and CUDA toolkit
- PyTorch with CUDA support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [WongKinYiu](https://github.com/WongKinYiu) for YOLOv7/YOLOv9 