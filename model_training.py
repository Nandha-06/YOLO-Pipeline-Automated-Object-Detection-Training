import os
import argparse
import json
import tempfile
import subprocess
import sys
from pathlib import Path
import platform

# Global variables
monitoring = False

# YOLO model repositories
YOLO_REPOS = {
    "v5": "https://github.com/ultralytics/yolov5.git",
    "v6": "https://github.com/meituan/YOLOv6.git",
    "v7": "https://github.com/WongKinYiu/yolov7.git",
    "v8": "https://github.com/ultralytics/ultralytics.git",
    "v9": "https://github.com/WongKinYiu/yolov9.git",
    "v10": "https://github.com/ultralytics/ultralytics.git",
    "v11": "https://github.com/ultralytics/ultralytics.git"
}

def check_gpu():
    """Check if a GPU is available and return its details"""
    try:
        # Try to import torch to check GPU
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = torch.cuda.get_device_capability(0)
            
            print(f"\n====== GPU INFORMATION ======")
            print(f"GPU Available: Yes")
            print(f"Number of GPUs: {gpu_count}")
            print(f"GPU Name: {gpu_name}")
            print(f"CUDA Capability: {gpu_capability[0]}.{gpu_capability[1]}")
            print(f"PyTorch CUDA Version: {torch.version.cuda}")
            print(f"==============================\n")
            
            return True, gpu_name
        else:
            print("\n====== GPU INFORMATION ======")
            print("GPU Available: No")
            print("Training will use CPU (this will be much slower)")
            print("==============================\n")
            return False, None
            
    except ImportError:
        print("\nWarning: PyTorch not installed. Cannot check GPU availability.")
        print("Installing PyTorch with CUDA support is recommended for faster training.\n")
        return False, None
    except Exception as e:
        print(f"\nError checking GPU: {e}")
        return False, None

def get_last_extraction_path():
    """Get the last extraction path saved by data_ingestion.py"""
    try:
        temp_dir = tempfile.gettempdir()
        path_file = os.path.join(temp_dir, 'yolo_extraction_path.json')
        
        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                data = json.load(f)
                path = data.get('last_extraction_path')
                if path and os.path.exists(path):
                    print(f"Using dataset directory: {path}")
                    return path
    except Exception as e:
        print(f"Error reading extraction path: {str(e)}")
    
    return None

def run_command(cmd, verbose=True):
    """Run a command and return its success status"""
    try:
        if verbose:
            # Run the command and show output in real-time
            result = subprocess.run(cmd, shell=True, check=False)
            return result.returncode == 0
        else:
            # Run silently
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def get_yolo_version():
    """Prompt the user to select a YOLO version"""
    print("Available YOLO versions:")
    available_versions = list(YOLO_REPOS.keys())
    for i, v in enumerate(available_versions, 1):
        # Display as YOLOv5, YOLOv6, etc.
        print(f"{i}. YOLOv{v[1:]}")

    while True:
        version_input = input("Enter YOLO version (5-11 or full number, default=8): ").strip().lower()
        
        # Default to v8 if empty input
        if not version_input:
            return "v8"
        
        # Handle format with or without 'v' prefix
        if version_input.startswith('v'):
            version = version_input
        else:
            version = f"v{version_input}"
            
        if version in YOLO_REPOS:
            return version
        else:
            print(f"Invalid version. Please enter a version between 5-11.")
            print("Defaulting to YOLOv8.")
            return "v8"

def get_model_size():
    """Prompt the user to select a model size"""
    print("\nAvailable model sizes:")
    sizes = {
        "n": "Nano - Fastest, lowest memory usage, less accurate",
        "s": "Small - Fast with good accuracy (default)",
        "m": "Medium - Balanced speed and accuracy",
        "l": "Large - Higher accuracy, slower inference",
        "x": "XLarge - Highest accuracy, slowest inference"
    }
    
    for size, description in sizes.items():
        print(f"{size}: {description}")
    
    while True:
        size_input = input("\nSelect model size (n/s/m/l/x, default=s): ").strip().lower()
        
        # Print the input for debugging
        print(f"Debug - You entered: '{size_input}'")
        
        # Default to 's' if empty input
        if not size_input:
            print("Using default size: s (Small)")
            return "s"
            
        if size_input in sizes:
            print(f"Selected size: {size_input} ({sizes[size_input].split(' - ')[0]})")
            return size_input
        else:
            print(f"Invalid size '{size_input}'. Please enter one of: {', '.join(sizes.keys())}")

def monitor_gpu_usage():
    """Monitor GPU usage during training"""
    global monitoring
    try:
        import torch
        import threading
        import time
        import psutil
        
        def gpu_monitor():
            """Monitor GPU usage in background thread"""
            print("\nGPU Monitoring Started...")
            while monitoring:
                try:
                    # Get GPU memory stats
                    gpu_stats = []
                    for i in range(torch.cuda.device_count()):
                        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        free = total_mem - reserved
                        gpu_stats.append((i, allocated, reserved, free, total_mem))
                    
                    # Get CPU and system memory
                    cpu_percent = psutil.cpu_percent()
                    ram_percent = psutil.virtual_memory().percent
                    
                    print(f"\r[Monitor] CPU: {cpu_percent:3.1f}% | RAM: {ram_percent:3.1f}%", end="")
                    
                    # Print GPU stats for each GPU
                    for i, allocated, reserved, free, total in gpu_stats:
                        print(f" | GPU {i}: {allocated:.2f}GB/{total:.2f}GB", end="")
                    
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    print(f"\nGPU monitoring error: {e}")
                    break
        
        # Set up monitoring flag and thread
        monitoring = True
        monitor_thread = threading.Thread(target=gpu_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    except ImportError:
        print("Could not import required packages for GPU monitoring")
        return None
    except Exception as e:
        print(f"Error setting up GPU monitoring: {e}")
        return None

def ensure_gpu_compatibility():
    """Ensure CUDA is correctly configured for PyTorch"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("\nWARNING: CUDA not available. Install CUDA and PyTorch with CUDA support.")
            return False
        
        # Try a simple CUDA operation to verify everything works
        try:
            x = torch.rand(10, 10).cuda()
            y = torch.rand(10, 10).cuda()
            z = x + y  # Simple CUDA operation
            del x, y, z
            torch.cuda.empty_cache()
            print("\nCUDA test successful!")
            return True
        except Exception as e:
            print(f"\nCUDA test failed: {e}")
            print("Your CUDA installation may be incompatible with PyTorch.")
            return False
    except ImportError:
        print("\nWarning: PyTorch not installed. Cannot check CUDA compatibility.")
        return False

def train_model():
    global monitoring
    parser = argparse.ArgumentParser(description='YOLO model training')
    parser.add_argument('--data_dir', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', help='Pre-trained weights file')
    parser.add_argument('--size', choices=['n', 's', 'm', 'l', 'x'], help='Model size (if not specified, will prompt)')
    parser.add_argument('--device', help='Device to train on (e.g., 0 for GPU, cpu for CPU)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--force-gpu', action='store_true', help='Force GPU usage even if checks fail')
    parser.add_argument('--monitor-gpu', action='store_true', help='Monitor GPU usage during training')
    
    args = parser.parse_args()
    
    # First, check for GPU and test CUDA compatibility
    has_gpu, gpu_name = check_gpu()
    
    if has_gpu:
        cuda_ok = ensure_gpu_compatibility()
        if not cuda_ok and not args.force_gpu:
            print("\nWARNING: CUDA compatibility check failed.")
            use_gpu_anyway = input("Try to use GPU anyway? (y/n, default=n): ").strip().lower()
            if not use_gpu_anyway.startswith('y'):
                print("Falling back to CPU training (this will be much slower)")
                has_gpu = False
    
    # Always ask for YOLO version regardless of command line arguments
    version = get_yolo_version()
    
    # Ask for model size if not provided via command line
    if args.size:
        model_size = args.size
        print(f"Using command line model size: {model_size}")
    else:
        model_size = get_model_size()
    
    # Double-check the model size value
    print(f"Selected model size: {model_size.upper()}")
    size_confirm = input(f"Is '{model_size}' the correct size? (y/n): ").strip().lower()
    if not size_confirm.startswith('y'):
        print("Let's select the model size again.")
        model_size = get_model_size()
    
    # Get dataset directory
    data_dir = args.data_dir or get_last_extraction_path()
    if not data_dir:
        data_dir = input("Enter dataset directory path: ")
        if not os.path.exists(data_dir):
            print(f"Error: Directory not found: {data_dir}")
            return
    
    # Ask for number of epochs
    epochs_input = input(f"Enter number of epochs for training (default={args.epochs}): ").strip()
    if epochs_input and epochs_input.isdigit():
        epochs = int(epochs_input)
    else:
        print(f"Using default: {args.epochs} epochs")
        epochs = args.epochs
    
    # Optimize batch size based on model size and available GPU memory
    if has_gpu:
        import torch
        # Get available GPU memory in GB
        try:
            free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\nDetected {free_mem:.2f} GB total GPU memory")
            
            # Suggested batch sizes based on model size and GPU memory
            if model_size == 'x':
                suggested_batch = min(int(free_mem / 2), 8)  # XLarge uses ~2GB per image
            elif model_size == 'l':
                suggested_batch = min(int(free_mem / 1.5), 12)  # Large uses ~1.5GB per image
            elif model_size == 'm':
                suggested_batch = min(int(free_mem / 1), 16)  # Medium uses ~1GB per image
            elif model_size == 's':
                suggested_batch = min(int(free_mem / 0.6), 24)  # Small uses ~0.6GB per image
            else:  # nano
                suggested_batch = min(int(free_mem / 0.3), 32)  # Nano uses ~0.3GB per image
            
            # Ensure batch size is at least 1
            suggested_batch = max(1, suggested_batch)
            
            if suggested_batch < args.batch:
                print(f"Warning: Your GPU may not have enough memory for batch size {args.batch}")
                print(f"Suggested batch size for your GPU: {suggested_batch}")
                use_suggested = input(f"Use suggested batch size of {suggested_batch}? (y/n, default=y): ").strip().lower()
                if not use_suggested.startswith('n'):
                    batch_size = suggested_batch
                    print(f"Using batch size: {batch_size}")
                else:
                    batch_size = args.batch
                    print(f"Using requested batch size: {batch_size} (may cause out of memory errors)")
            else:
                batch_size = args.batch
        except Exception as e:
            print(f"Error estimating batch size: {e}")
            batch_size = args.batch
    else:
        # CPU training - use small batch size
        batch_size = min(args.batch, 4)
        print(f"CPU training: Using smaller batch size of {batch_size}")
    
    # Clone repository if needed
    repo_dir = f"yolo{version}"
    if not os.path.exists(repo_dir):
        print(f"Cloning YOLOv{version[1:]} repository...")
        clone_cmd = f"git clone {YOLO_REPOS[version]} {repo_dir}"
        success = run_command(clone_cmd, verbose=True)
        if not success:
            print("Repository cloning failed.")
            return
        
        # Check if directory exists after cloning to confirm success
        if os.path.exists(repo_dir):
            print(f"Repository cloned successfully to {repo_dir}")
        else:
            print(f"Repository directory {repo_dir} not found after cloning.")
            return
    else:
        print(f"Using existing {repo_dir} repository.")
    
    # Install requirements
    req_file = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing requirements...")
        run_command(f"{sys.executable} -m pip install -r {req_file}", verbose=False)
        print("Requirements installed.")
    
    # Make sure we have GPU monitoring package if needed
    if args.monitor_gpu or has_gpu:
        try:
            import psutil
        except ImportError:
            print("Installing psutil for system monitoring...")
            run_command(f"{sys.executable} -m pip install psutil", verbose=False)
    
    # Find data.yaml file
    yaml_file = next((str(f) for f in Path(data_dir).glob("*.yaml")), None)
    if not yaml_file:
        yaml_file = os.path.join(data_dir, "data.yaml")
        print(f"No YAML config found. Please create {yaml_file} manually.")
        return
    print(f"Using dataset config: {yaml_file}")
    
    # Determine weights file
    weights = args.weights
    if not weights:
        # Default weights based on version and size
        if version == "v5":
            weights = f"yolov5{model_size}.pt"
        elif version == "v7":
            # YOLOv7 has different naming convention
            if model_size == "s":
                weights = "yolov7.pt"
            elif model_size == "m":
                weights = "yolov7-e6.pt" 
            elif model_size == "l":
                weights = "yolov7-d6.pt"
            elif model_size == "x":
                weights = "yolov7-e6e.pt"
            else:  # nano
                weights = "yolov7-tiny.pt"
        elif version == "v9":
            # YOLOv9 naming convention
            if model_size == "s":
                weights = "yolov9c.pt"
            elif model_size == "m": 
                weights = "yolov9e.pt"
            elif model_size in ["l", "x"]:
                weights = "yolov9m.pt"  # largest available
            else:  # nano
                weights = "yolov9n.pt"
        elif version in ["v8", "v10", "v11"]:
            weights = f"yolo{version}{model_size}.pt"
        else:
            weights = ""
    
    # Create and run training command
    print(f"\nPreparing to train YOLOv{version[1:]} {model_size.upper()} model...")
    
    # Quote paths to handle spaces in Windows
    yaml_file_quoted = f'"{yaml_file}"'
    
    # Set device parameter
    device_param = ""
    if args.device:
        device_param = f"device={args.device}"
    elif has_gpu:
        device_param = "device=0"  # Use first GPU by default
    else:
        device_param = "device=cpu"
    
    # Set additional parameters for better GPU usage
    if has_gpu:
        # Add cache parameter to speed up training
        cache_param = "cache=True"
        # Reduce image augmentation to speed up processing
        mosaic_param = "mosaic=1"
        # Add mixed precision training for faster GPU computation
        mixed_precision_param = "half=True"
    else:
        cache_param = ""
        mosaic_param = ""
        mixed_precision_param = ""
    
    # Set verbose flag
    verbose_param = "verbose=True" if args.verbose or not has_gpu else ""
    
    if version in ["v8", "v10", "v11"]:
        # Ultralytics API
        print("Installing ultralytics...")
        run_command(f"{sys.executable} -m pip install ultralytics", verbose=False)
        
        cmd = (f"yolo train "
               f"model={weights} data={yaml_file_quoted} epochs={epochs} "
               f"imgsz={args.img_size} batch={batch_size} {device_param} "
               f"workers={0 if platform.system() == 'Windows' else 8} {verbose_param} "
               f"{cache_param} {mosaic_param} {mixed_precision_param}")
    else:
        # Traditional YOLO training
        train_script = os.path.join(repo_dir, "train.py")
        if not os.path.exists(train_script):
            train_script = os.path.join(repo_dir, "tools", "train.py")
            if not os.path.exists(train_script):
                print(f"Cannot find training script for YOLOv{version}")
                return
        
        device_arg = f"--device {args.device}" if args.device else f"--device 0" if has_gpu else "--device cpu"
        workers_arg = "--workers 0" if platform.system() == 'Windows' else "--workers 8"
        
        # Add GPU-specific parameters for YOLOv5/7
        gpu_specific = ""
        if has_gpu:
            gpu_specific = "--cache --half"
        
        cmd = (f"{sys.executable} {train_script} "
               f"--data {yaml_file_quoted} --epochs {epochs} "
               f"--batch-size {batch_size} --img-size {args.img_size} "
               f"{device_arg} {workers_arg} {gpu_specific}")
        
        if weights:
            cmd += f" --weights {weights}"
        
        if args.verbose or not has_gpu:
            cmd += " --verbose"
    
    # Ask for confirmation
    print("\nTraining command:")
    print(cmd)
    confirm = input("\nStart training? (y/n): ").lower()
    
    if confirm.startswith('y'):
        print("\nStarting training...")
        
        # Start GPU monitoring if requested
        monitor_thread = None
        if has_gpu and (args.monitor_gpu or True):  # Always monitor by default
            monitor_thread = monitor_gpu_usage()
        
        print("Training logs will be displayed below (this may take a while)...")
        print("-" * 60)
        
        # Run the training command
        try:
            success = run_command(cmd, verbose=True)
            
            # Stop GPU monitoring
            if monitor_thread:
                monitoring = False
                monitor_thread.join(timeout=1)
                
            if success:
                print("\nTraining complete!")
            else:
                print("\nTraining failed. Check the error messages above.")
                
                if not has_gpu:
                    print("\nSuggestion: Training without a GPU can be very slow.")
                    print("Consider using a cloud GPU service like Google Colab if training fails on your machine.")
                elif "CUDA out of memory" in cmd or "GPU memory" in cmd:
                    print("\nYour GPU ran out of memory. Try the following:")
                    print("1. Reduce batch size (--batch)")
                    print("2. Reduce image size (--img_size)")
                    print("3. Use a smaller model (n or s instead of m, l, or x)")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
            if monitor_thread:
                monitoring = False
                monitor_thread.join(timeout=1)
    else:
        print("\nTraining aborted.")
        print(f"You can run the command manually: {cmd}")

if __name__ == "__main__":
    train_model() 