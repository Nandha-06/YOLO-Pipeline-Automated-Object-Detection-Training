# YOLO Model Training Template

Simple tools for YOLO dataset preparation, validation, and training.

## Install
```bash
pip install -r requirements.txt
```

## Data Ingestion

Download files and folders from Google Drive using gdown.

### Interactive Mode
```bash
python data_ingestion.py
```
The script will prompt you for a Google Drive URL and automatically validate the downloaded dataset.

### Command Line Mode
#### Download a file
```bash
python data_ingestion.py "https://drive.google.com/file/d/FILEID/view"
```

#### Download a file with custom output path
```bash
python data_ingestion.py "https://drive.google.com/file/d/FILEID/view" -o "data/file.zip"
```

#### Download a folder
```bash
python data_ingestion.py "https://drive.google.com/drive/folders/FOLDERID" -f
```

#### Download multiple files
```bash
python data_ingestion.py "URL1" "URL2" "URL3"
```

#### Download without extracting zip files
```bash
python data_ingestion.py "URL" -n
```

#### Download and create YAML if missing
```bash
python data_ingestion.py "URL" -y
```

#### Keep zip files after extraction
```bash
python data_ingestion.py "URL" -k
```

### Options
- `-q` Quiet mode
- `-o` Output path
- `-f` Folder mode
- `-n` No extraction (don't extract zip files)
- `-v` Validate dataset after downloading
- `-y` Create YAML template if missing (disabled by default)
- `-k` Keep zip files after extraction (by default, zip files are deleted)

## Data Validation

Validate if your dataset has the necessary files and structure for YOLO training.

### Usage
```bash
python data_validation.py
```
Running without arguments will automatically use the last extracted directory from data_ingestion.py.

If you want to specify a different directory:
```bash
python data_validation.py "path/to/dataset"
```

#### Create a YAML template if missing
```bash
python data_validation.py --create-yaml
```

### What it validates:
- Dataset directory structure
- Presence of YAML configuration file
- Image files in expected locations
- Label files in expected locations

The script provides a summary of validation results and recommendations for fixing issues.

## Model Training

Train a YOLO model using the validated dataset.

### Supported YOLO Versions

The training script supports multiple YOLO versions:

- **YOLOv5** - Reliable and widely used
- **YOLOv6** - Efficient industrial object detector by Meituan
- **YOLOv7** - State-of-the-art object detector
- **YOLOv8** - Latest version with segmentation support
- **YOLOv9** - Latest from WongKinYiu with GELAN architecture 
- **YOLOv10** - From Ultralytics with improved performance
- **YOLO11** - Latest from Ultralytics, most advanced
- **YOLOX** - Through super-gradients framework

### Interactive Mode
```bash
python model_training.py
```
The script will prompt you to choose a YOLO version and set hyperparameters.

### Command Line Mode
```bash
python model_training.py --version v8 --epochs 100 --batch_size 16 --img_size 640
```

### Options
- `--version` YOLO version to use (v5, v6, v7, v8, v9, v10, v11, x)
- `--data_dir` Path to dataset directory (uses last extracted directory by default)
- `--epochs` Number of training epochs
- `--batch_size` Batch size for training
- `--img_size` Input image size
- `--lr` Learning rate
- `--force_clone` Force re-cloning the repository if it exists
- `--interactive` Use interactive mode

### What it does:
1. Clones the selected YOLO repository from GitHub
2. Installs required dependencies
3. Finds or creates the dataset YAML configuration
4. Sets up training with specified hyperparameters
5. Runs the training process

## Complete Workflow Example

```bash
# Step 1: Download dataset from Google Drive, extract it, and delete the zip file
python data_ingestion.py "https://drive.google.com/file/d/FILEID/view"

# Step 2: Run validation (uses last extracted directory automatically)
python data_validation.py

# Step 3: Create YAML if needed
python data_validation.py --create-yaml

# Step 4: Train YOLO model (uses the same dataset directory)
python model_training.py --version v11 --epochs 100
``` 