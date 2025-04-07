# YOLO Workflow Automation

A comprehensive Python-based workflow for YOLO (You Only Look Once) object detection model training. This project provides an end-to-end solution for downloading, validating, and training YOLO models with minimal user intervention.

## Features

- **Data Ingestion**: Download datasets from Google Drive with automatic extraction and validation
- **Data Validation**: Comprehensive dataset validation for YOLO training requirements
- **Model Training**: Support for multiple YOLO versions (v5-v11) with GPU optimization
- **Automated Workflow**: Single-command execution of the entire training pipeline
- **GPU Monitoring**: Real-time GPU usage monitoring during training
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Google Drive access (for dataset downloads)
- Basic understanding of YOLO object detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nandha-06/YOLO-Pipeline-Automated-Object-Detection-Training.git
cd YOLO-Pipeline-Automated-Object-Detection-Training
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data_ingestion.py    # Handles dataset downloading and extraction
├── data_validation.py   # Validates dataset structure and content
├── model_training.py    # Manages YOLO model training process
├── main.py             # Main workflow orchestrator
└── README.md           # Project documentation
```

## Usage

### Basic Usage

Run the complete workflow with a single command:
```bash
python main.py
```

The workflow will guide you through:
1. Dataset download from Google Drive
2. Dataset validation
3. Model training configuration
4. Training execution

### Advanced Usage

#### Data Ingestion
```bash
python data_ingestion.py [options]
```

Options:
- `--urls`: Google Drive URLs or file IDs
- `--output`: Output path for downloaded files
- `--folder`: Download entire folder
- `--quiet`: Suppress download progress
- `--no-extract`: Do not extract zip files
- `--validate`: Validate dataset after downloading
- `--create-yaml`: Create YAML template if missing
- `--keep-zip`: Keep zip files after extraction

#### Data Validation
```bash
python data_validation.py [data_dir] [options]
```

Options:
- `--create-yaml`: Create a template YAML file if missing

#### Model Training
```bash
python model_training.py [options]
```

Options:
- `--data_dir`: Dataset directory
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--img_size`: Image size
- `--weights`: Pre-trained weights file
- `--size`: Model size (n/s/m/l/x)
- `--device`: Device to train on (e.g., 0 for GPU, cpu for CPU)
- `--verbose`: Enable verbose output
- `--force-gpu`: Force GPU usage even if checks fail
- `--monitor-gpu`: Monitor GPU usage during training
- `--project`: Project directory for saving results
- `--name`: Experiment name

## Supported YOLO Versions

- YOLOv5
- YOLOv6
- YOLOv7
- YOLOv8
- YOLOv9
- YOLOv10
- YOLOv11

## Dataset Requirements

The workflow expects datasets in YOLO format with the following structure:
```
dataset/
├── images/           # Training images
├── labels/           # Label files
├── train/            # Training split
├── valid/            # Validation split
└── test/             # Test split (optional)
```

## GPU Requirements

- CUDA-compatible GPU
- NVIDIA drivers
- PyTorch with CUDA support
- Minimum 4GB VRAM (8GB recommended)

## Troubleshooting

1. **GPU Not Detected**
   - Ensure CUDA is properly installed
   - Check PyTorch CUDA version matches system CUDA version
   - Verify NVIDIA drivers are up to date

2. **Dataset Validation Failures**
   - Check dataset structure matches YOLO requirements
   - Ensure image and label files are properly paired
   - Verify YAML configuration file exists and is valid

3. **Training Issues**
   - Reduce batch size if out of memory
   - Try smaller model size
   - Check dataset quality and size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- Ultralytics for YOLOv5 and YOLOv8
- YOLOv6, YOLOv7, YOLOv9, YOLOv10, and YOLOv11 teams
- PyTorch team for the deep learning framework
