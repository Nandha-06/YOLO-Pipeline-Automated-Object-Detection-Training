import os
import glob
import yaml
import argparse
import json
import tempfile
from pathlib import Path

def get_last_extraction_path():
    """
    Get the last extraction path saved by data_ingestion.py
    
    Returns:
        str: Path to the last extracted directory or None if not found
    """
    try:
        # Look for the path file in the temp directory
        temp_dir = tempfile.gettempdir()
        path_file = os.path.join(temp_dir, 'yolo_extraction_path.json')
        
        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                data = json.load(f)
                extraction_path = data.get('last_extraction_path')
                
                if extraction_path and os.path.exists(extraction_path):
                    print(f"Using last extracted directory: {extraction_path}")
                    return extraction_path
    except Exception as e:
        print(f"Error reading last extraction path: {str(e)}")
    
    return None

def validate_directory_structure(data_dir):
    """
    Validate if the directory has the expected YOLO dataset structure
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        dict: Dictionary with validation results
    """
    results = {
        "valid": True,
        "missing_dirs": [],
        "found_dirs": []
    }
    
    # Common directory names in YOLO datasets
    expected_dirs = ['images', 'labels', 'train', 'valid', 'test']
    
    # Check if at least one of the expected directories exists
    found_any = False
    
    for dir_name in expected_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.isdir(dir_path):
            results["found_dirs"].append(dir_name)
            found_any = True
        else:
            # Also check for nested structures like train/images, valid/labels, etc.
            nested_found = False
            for parent in ['train', 'valid', 'test']:
                nested_path = os.path.join(data_dir, parent, dir_name)
                if os.path.isdir(nested_path):
                    results["found_dirs"].append(f"{parent}/{dir_name}")
                    found_any = True
                    nested_found = True
            
            if not nested_found:
                results["missing_dirs"].append(dir_name)
    
    if not found_any:
        results["valid"] = False
        results["error"] = "No expected dataset directories found"
        
    return results

def validate_yaml_config(data_dir):
    """
    Check if there's a YAML config file for the dataset
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        dict: Dictionary with validation results
    """
    results = {
        "valid": False,
        "yaml_files": []
    }
    
    # Look for YAML files
    yaml_files = glob.glob(os.path.join(data_dir, "*.yaml")) + glob.glob(os.path.join(data_dir, "*.yml"))
    
    if yaml_files:
        results["valid"] = True
        results["yaml_files"] = [os.path.basename(f) for f in yaml_files]
        
        # Try to load the first YAML file to validate its structure
        try:
            with open(yaml_files[0], 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            # Check for typical YOLO dataset YAML keys
            expected_keys = ['nc', 'names', 'train', 'val', 'test']
            missing_keys = [key for key in expected_keys if key not in yaml_data]
            
            if missing_keys:
                results["warning"] = f"YAML file is missing recommended keys: {', '.join(missing_keys)}"
                
        except Exception as e:
            results["warning"] = f"YAML file exists but could not be parsed: {str(e)}"
    
    return results

def validate_image_files(data_dir):
    """
    Check if there are image files in the dataset
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        dict: Dictionary with validation results
    """
    results = {
        "valid": False,
        "image_count": 0,
        "image_dirs": []
    }
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    # Look for images in common directories
    image_dirs = []
    total_images = 0
    
    # Check in images/ directory
    images_dir = os.path.join(data_dir, 'images')
    if os.path.isdir(images_dir):
        image_count = count_files_with_extensions(images_dir, image_extensions)
        if image_count > 0:
            image_dirs.append('images')
            total_images += image_count
    
    # Check in train/images, valid/images, etc.
    for split in ['train', 'valid', 'test']:
        split_img_dir = os.path.join(data_dir, split, 'images')
        if os.path.isdir(split_img_dir):
            image_count = count_files_with_extensions(split_img_dir, image_extensions)
            if image_count > 0:
                image_dirs.append(f"{split}/images")
                total_images += image_count
    
    # Check for flat structure
    if not image_dirs:
        image_count = count_files_with_extensions(data_dir, image_extensions)
        if image_count > 0:
            image_dirs.append('.')
            total_images += image_count
    
    results["image_count"] = total_images
    results["image_dirs"] = image_dirs
    
    if total_images > 0:
        results["valid"] = True
    
    return results

def validate_label_files(data_dir):
    """
    Check if there are label files in the dataset
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        dict: Dictionary with validation results
    """
    results = {
        "valid": False,
        "label_count": 0,
        "label_dirs": []
    }
    
    # YOLO label files are typically .txt
    label_extensions = ['.txt']
    
    # Look for labels in common directories
    label_dirs = []
    total_labels = 0
    
    # Check in labels/ directory
    labels_dir = os.path.join(data_dir, 'labels')
    if os.path.isdir(labels_dir):
        label_count = count_files_with_extensions(labels_dir, label_extensions)
        if label_count > 0:
            label_dirs.append('labels')
            total_labels += label_count
    
    # Check in train/labels, valid/labels, etc.
    for split in ['train', 'valid', 'test']:
        split_label_dir = os.path.join(data_dir, split, 'labels')
        if os.path.isdir(split_label_dir):
            label_count = count_files_with_extensions(split_label_dir, label_extensions)
            if label_count > 0:
                label_dirs.append(f"{split}/labels")
                total_labels += label_count
    
    results["label_count"] = total_labels
    results["label_dirs"] = label_dirs
    
    if total_labels > 0:
        results["valid"] = True
    
    return results

def count_files_with_extensions(directory, extensions):
    """
    Count files with specific extensions in a directory
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to count
        
    Returns:
        int: Number of files with the given extensions
    """
    count = 0
    for ext in extensions:
        count += len(glob.glob(os.path.join(directory, f"*{ext}")))
    return count

def validate_dataset(data_dir):
    """
    Validate if the directory contains a valid YOLO dataset
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        dict: Dictionary with validation results
    """
    # Make sure the path exists
    if not os.path.exists(data_dir):
        return {
            "valid": False,
            "error": f"Directory does not exist: {data_dir}"
        }
    
    # Run all validation checks
    results = {
        "directory": data_dir,
        "structure": validate_directory_structure(data_dir),
        "yaml_config": validate_yaml_config(data_dir),
        "images": validate_image_files(data_dir),
        "labels": validate_label_files(data_dir)
    }
    
    # Overall validation result
    results["valid"] = results["images"]["valid"] and \
                       (results["labels"]["valid"] or len(results["structure"]["found_dirs"]) > 0)
    
    return results

def print_validation_results(results):
    """
    Print validation results in a user-friendly format
    
    Args:
        results: Dictionary with validation results
    """
    print("\n===== DATASET VALIDATION RESULTS =====")
    print(f"Directory: {results['directory']}")
    print(f"Overall Status: {'✓ Valid' if results['valid'] else '✗ Invalid'}")
    
    print("\nDirectory Structure:")
    if results["structure"]["found_dirs"]:
        print(f"  ✓ Found directories: {', '.join(results['structure']['found_dirs'])}")
    else:
        print("  ✗ No expected directories found")
    
    print("\nYAML Configuration:")
    if results["yaml_config"]["valid"]:
        print(f"  ✓ Found YAML files: {', '.join(results['yaml_config']['yaml_files'])}")
        if "warning" in results["yaml_config"]:
            print(f"  ⚠ {results['yaml_config']['warning']}")
    else:
        print("  ✗ No YAML configuration files found")
    
    print("\nImages:")
    if results["images"]["valid"]:
        print(f"  ✓ Found {results['images']['image_count']} images in: {', '.join(results['images']['image_dirs'])}")
    else:
        print("  ✗ No image files found")
    
    print("\nLabels:")
    if results["labels"]["valid"]:
        print(f"  ✓ Found {results['labels']['label_count']} label files in: {', '.join(results['labels']['label_dirs'])}")
    else:
        print("  ✗ No label files found")
    
    print("\nRecommendations:")
    if not results["valid"]:
        if not results["images"]["valid"]:
            print("  - Add image files in an 'images' directory or train/valid/test splits")
        if not results["labels"]["valid"]:
            print("  - Add label files in a 'labels' directory or train/valid/test splits")
        if not results["yaml_config"]["valid"]:
            print("  - Create a YAML configuration file with dataset information")
    else:
        print("  ✓ Dataset looks valid for YOLO training!")

def create_yaml_template(data_dir, class_names=None):
    """
    Create a template YAML file for the dataset
    
    Args:
        data_dir: Path to the data directory
        class_names: List of class names (optional)
        
    Returns:
        str: Path to the created YAML file
    """
    if class_names is None:
        class_names = ['class1', 'class2', 'class3']  # Placeholder
    
    yaml_data = {
        'nc': len(class_names),
        'names': class_names,
        'train': './train/images',
        'val': './valid/images',
        # 'test' key is optional but recommended
    }
    
    yaml_path = os.path.join(data_dir, 'data.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created YAML template at: {yaml_path}")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description='Validate YOLO dataset structure')
    parser.add_argument('data_dir', nargs='?', help='Path to the dataset directory')
    parser.add_argument('--create-yaml', action='store_true', help='Create a template YAML file if missing')
    
    args = parser.parse_args()
    
    # If no directory provided:
    # 1. Try to use the last extraction path
    # 2. If that doesn't exist, use current directory
    if not args.data_dir:
        data_dir = get_last_extraction_path() or os.getcwd()
    else:
        data_dir = args.data_dir
        
    # Validate the dataset
    results = validate_dataset(data_dir)
    print_validation_results(results)
    
    # Create YAML template if requested
    if args.create_yaml and not results["yaml_config"]["valid"]:
        create_yaml_template(data_dir)
    
    return results["valid"]

if __name__ == "__main__":
    main() 