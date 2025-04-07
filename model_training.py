#!/usr/bin/env python
import os
import argparse
import json
import tempfile
import subprocess
import sys
from pathlib import Path
import platform
import importlib.util
import re

# Force CUDA device selection and compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force using the first GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Prevent memory fragmentation
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Make CUDA errors synchronous for easier debugging

# For PyTorch compatibility with CUDA driver/toolkit mismatch
if importlib.util.find_spec("torch") is not None:
    import torch
    if torch.cuda.is_available():
        # Force PyTorch to initialize the GPU
        _ = torch.zeros(1).cuda()
        print(f"\n✅ Successfully initialized GPU: {torch.cuda.get_device_name(0)}")
        print(f"   PyTorch CUDA: {torch.version.cuda}, Device capability: {torch.cuda.get_device_capability(0)}")
        # Print memory info to confirm GPU is accessible
        print(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

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

# YOLO model weight URLs - for direct downloads
YOLO_WEIGHTS = {
    "v5": {
        "n": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
        "s": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        "m": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
        "l": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
        "x": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"
    },
    "v6": {
        "n": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt",
        "s": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt",
        "m": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt",
        "l": "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt"
    },
    "v7": {
        "n": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
        "s": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
        "m": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt",
        "l": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt",
        "x": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt"
    },
    "v8": {
        "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"
    },
    "v9": {
        "n": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-n.pt",
        "s": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt",
        "m": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt",
        "l": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt",
        "x": "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt"
    },
    # For v10 and v11, we'll use the v8 weights as they're based on the same ultralytics codebase
    "v10": {
        "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"
    },
    "v11": {
        "n": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "s": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "m": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "l": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "x": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"
    }
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
        
        # Check if input is a list number (1-7)
        if version_input.isdigit() and 1 <= int(version_input) <= len(available_versions):
            # Convert list selection (1-7) to actual version (v5-v11)
            index = int(version_input) - 1
            version = available_versions[index]
            print(f"Selected version from menu: YOLOv{version[1:]} (option {version_input})")
            return version
        
        # Handle format with or without 'v' prefix
        if version_input.startswith('v'):
            version = version_input
        else:
            version = f"v{version_input}"
            
        if version in YOLO_REPOS:
            return version
        else:
            print(f"Invalid version. Please enter a number between 1-{len(available_versions)} or a version between 5-11.")
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
        import os
        
        def gpu_monitor():
            """Monitor GPU usage in background thread"""
            print("\nGPU Monitoring Started...")
            
            # Check if we're using Windows
            is_windows = os.name == 'nt'
            
            # For non-Windows systems, try to use nvidia-smi directly for more accurate readings
            nvml_initialized = False
            if not is_windows:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    nvml_initialized = True
                    print("Using NVML for GPU monitoring")
                except ImportError:
                    print("NVML not available, using PyTorch for GPU monitoring")
                except Exception as e:
                    print(f"Error initializing NVML: {e}")
            
            # On Windows, try to use pynvml instead
            if is_windows:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    nvml_initialized = True
                    print("Using NVML for GPU monitoring on Windows")
                except ImportError:
                    print("Installing pynvml for better GPU monitoring...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "pynvml"])
                        import pynvml
                        pynvml.nvmlInit()
                        nvml_initialized = True
                        print("Using NVML for GPU monitoring")
                    except Exception as e:
                        print(f"NVML installation failed, using PyTorch for GPU monitoring: {e}")
                except Exception as e:
                    print(f"Error initializing NVML: {e}")
            
            peak_memory_used = 0
            last_active = False
            inactive_count = 0
            
            print("\nGPU MONITORING: If you don't see GPU memory usage increase, the GPU is not being used!")
            
            while monitoring:
                try:
                    # Get GPU memory stats
                    gpu_stats = []
                    gpu_utilization = []
                    for i in range(torch.cuda.device_count()):
                        # Get memory info
                        if nvml_initialized:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            total_mem = mem_info.total / 1024**3  # GB
                            used_mem = mem_info.used / 1024**3    # GB
                            free_mem = mem_info.free / 1024**3    # GB
                            gpu_stats.append((i, used_mem, used_mem, free_mem, total_mem))
                            
                            # Get GPU utilization
                            try:
                                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_util = util.gpu
                                mem_util = util.memory
                                gpu_utilization.append((gpu_util, mem_util))
                            except:
                                gpu_utilization.append((0, 0))
                            
                            # Update peak memory
                            if used_mem > peak_memory_used:
                                peak_memory_used = used_mem
                        else:
                            # Use PyTorch's memory tracking (less accurate)
                            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                            reserved = torch.cuda.memory_reserved(i) / 1024**3
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            free = total_mem - reserved
                            gpu_stats.append((i, allocated, reserved, free, total_mem))
                            
                            # Update peak memory
                            if allocated > peak_memory_used:
                                peak_memory_used = allocated
                    
                    # Get CPU and system memory
                    cpu_percent = psutil.cpu_percent()
                    ram_percent = psutil.virtual_memory().percent
                    
                    # Check if GPU is active (detect if memory usage is significant)
                    if gpu_stats:
                        is_active = gpu_stats[0][1] > 0.1  # Consider active if using > 100MB
                        
                        if is_active:
                            inactive_count = 0
                            if not last_active:
                                print("\n⚠️ GPU ACTIVATED! Now using GPU for computation.")
                        else:
                            inactive_count += 1
                            if inactive_count == 10 and last_active:  # 20 seconds of inactivity
                                print("\n⚠️ GPU INACTIVE! Model may have fallen back to CPU processing.")
                        
                        last_active = is_active
                    
                    # Clear previous line and print stats
                    print("\r" + " " * 100, end="\r")
                    print(f"\r[Monitor] CPU: {cpu_percent:3.1f}% | RAM: {ram_percent:3.1f}% | Peak GPU: {peak_memory_used:.2f}GB", end="")
                    
                    # Print GPU stats for each GPU
                    for idx, ((i, allocated, reserved, free, total), (gpu_util, mem_util)) in enumerate(zip(gpu_stats, gpu_utilization)):
                        active_marker = "✓" if allocated > 0.1 else "✗"
                        print(f" | GPU {i} [{active_marker}]: {allocated:.2f}GB/{total:.2f}GB ({gpu_util}%)", end="")
                    
                    # Explicitly flush the output to ensure real-time updates
                    sys.stdout.flush()
                    
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"\nGPU monitoring error: {e}")
                    time.sleep(5)  # Wait longer on error
            
            # Final summary when monitoring ends
            print(f"\n\nTraining complete! Peak GPU memory usage: {peak_memory_used:.2f}GB")
            
            # Clean up NVML if used
            if nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
        
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
        
        # Check CUDA version compatibility
        cuda_version_torch = torch.version.cuda
        print(f"\nPyTorch CUDA version: {cuda_version_torch}")
        
        # Get system CUDA version using subprocess
        try:
            import subprocess
            import re
            
            # Try nvidia-smi first (shows driver version)
            result = subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Extract CUDA version from nvidia-smi output
                match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
                if match:
                    cuda_version_driver = match.group(1)
                    print(f"System CUDA driver version: {cuda_version_driver}")
                    
                    # Try to get CUDA toolkit version from nvcc
                    try:
                        nvcc_result = subprocess.run('nvcc --version', shell=True, capture_output=True, text=True)
                        nvcc_match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
                        if nvcc_match:
                            cuda_version_toolkit = nvcc_match.group(1)
                            print(f"CUDA toolkit version: {cuda_version_toolkit}")
                            
                            # Now check if we have a mismatch situation
                            if cuda_version_toolkit != cuda_version_driver:
                                print(f"\n⚠️ DETECTED CUDA VERSION MISMATCH ⚠️")
                                print(f"  → Driver: {cuda_version_driver}, Toolkit: {cuda_version_toolkit}, PyTorch: {cuda_version_torch}")
                                print("  This mismatch can cause GPU utilization issues.")
                            
                            if cuda_version_torch != cuda_version_toolkit:
                                print(f"\nWARNING: PyTorch CUDA ({cuda_version_torch}) doesn't match toolkit ({cuda_version_toolkit}).")
                                print("This might cause GPU utilization problems. Consider reinstalling PyTorch:")
                                print(f"pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version_toolkit.replace('.', '')}")
                                
                                # Force GPU initialization to ensure compatibility
                                print("\nAttempting to initialize GPU regardless of version mismatch...")
                                try:
                                    # Force PyTorch to initialize the GPU
                                    _ = torch.zeros(1).cuda()
                                    device_props = torch.cuda.get_device_properties(0)
                                    print(f"\n✅ GPU initialized successfully: {torch.cuda.get_device_name(0)}")
                                    print(f"   GPU Compute Capability: {device_props.major}.{device_props.minor}")
                                    print(f"   Total GPU Memory: {device_props.total_memory / 1024**3:.2f} GB")
                                    return True
                                except Exception as e:
                                    print(f"❌ Failed to initialize GPU: {e}")
                                    return False
                    except Exception as e:
                        print(f"Could not check CUDA toolkit version: {e}")
        except Exception as e:
            print(f"Could not check system CUDA version: {e}")
            
        # If we got here without returning, try a simple GPU test
        try:
            # Force GPU initialization
            _ = torch.zeros(1).cuda()
            print(f"\n✅ GPU initialized successfully with PyTorch CUDA {cuda_version_torch}")
            print(f"   Device name: {torch.cuda.get_device_name(0)}")
            
            # Try a simple matrix multiplication to verify
            A = torch.rand(1000, 1000, device='cuda')
            B = torch.rand(1000, 1000, device='cuda')
            torch.matmul(A, B)
            torch.cuda.synchronize()  # Wait for GPU operation to complete
            print("   GPU computation test passed.")
            return True
        except Exception as e:
            print(f"\n❌ GPU initialization failed: {e}")
            print("   This might indicate a compatibility issue between PyTorch and your CUDA installation.")
            return False
            
    except ImportError:
        print("\nWarning: PyTorch not installed. Cannot check GPU availability.")
        print("Installing PyTorch with CUDA support is recommended for faster training.\n")
        return False
    except Exception as e:
        print(f"\nError checking GPU: {e}")
        return False

def verify_dataset_yaml(yaml_file, version):
    """Verify and fix dataset paths in YAML file for compatibility with the selected YOLO version"""
    import yaml
    
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the directory containing the YAML file
        yaml_dir = os.path.dirname(os.path.abspath(yaml_file))
        
        # Check if paths exist and adjust them if necessary
        modified = False
        
        # For YOLOv5, paths should be either absolute or relative to the YOLOv5 directory
        if version == "v5":
            # Check if train/val paths exist
            for key in ['train', 'val']:
                if key in data:
                    path = data[key]
                    # If path doesn't exist as specified, try to find it
                    if not os.path.exists(path):
                        # Try relative to yaml directory
                        rel_path = os.path.join(yaml_dir, path)
                        if os.path.exists(rel_path):
                            data[key] = os.path.abspath(rel_path)
                            modified = True
                            print(f"Fixed {key} path: {data[key]}")
                        else:
                            print(f"Warning: {key} path '{path}' not found")
            
            # Check if train/val image and label directories are used instead of list files
            if 'train' not in data and 'val' not in data:
                # For YOLOv5 structure, we need to specify the parent directory
                if all(key in data for key in ['path', 'names']):
                    dataset_path = data['path']
                    # Make path absolute if relative
                    if not os.path.isabs(dataset_path):
                        abs_path = os.path.abspath(os.path.join(yaml_dir, dataset_path))
                        data['path'] = abs_path
                        modified = True
                        print(f"Updated dataset path to absolute: {abs_path}")
        
        # Save modified YAML file if needed
        if modified:
            backup_file = yaml_file + '.backup'
            import shutil
            shutil.copy2(yaml_file, backup_file)
            print(f"Created backup of original YAML file: {backup_file}")
            
            with open(yaml_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Updated dataset configuration in {yaml_file}")
        
        return True
    except Exception as e:
        print(f"Error verifying dataset YAML: {e}")
        return False

def prepare_amp_check_models(version, model_size):
    """Prepare model files needed for AMP checks to prevent unnecessary downloads"""
    try:
        # For YOLOv8, v10, v11 the AMP check downloads a nano model regardless of selection
        if version in ["v8", "v10", "v11"]:
            # Define the paths for requested model and nano model
            selected_weights = f"yolo{version}{model_size}.pt"
            nano_weights = f"yolo{version}n.pt"
            yolo_weights = f"yolo{version[1:]}{model_size}.pt"  # Without the 'v' prefix
            yolo_nano = f"yolo{version[1:]}n.pt"  # Without the 'v' prefix
            
            print(f"Preparing AMP check model files to prevent unnecessary downloads...")
            
            # If we have the user-selected model but not the nano, copy it
            if os.path.exists(selected_weights) and not os.path.exists(nano_weights):
                print(f"Creating {nano_weights} from {selected_weights} to avoid additional downloads")
                import shutil
                shutil.copy2(selected_weights, nano_weights)
            
            # Also create files without the 'v' prefix that AMP checks sometimes look for
            if os.path.exists(selected_weights) and not os.path.exists(yolo_weights):
                import shutil
                shutil.copy2(selected_weights, yolo_weights)
                print(f"Created {yolo_weights} from {selected_weights}")
            
            if os.path.exists(nano_weights) and not os.path.exists(yolo_nano):
                import shutil
                shutil.copy2(nano_weights, yolo_nano)
                print(f"Created {yolo_nano} from {nano_weights}")
                
            return True
    except Exception as e:
        print(f"Warning: Failed to prepare AMP check models: {e}")
    
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
    parser.add_argument('--project', default='runs/train', help='Project directory for saving results')
    parser.add_argument('--name', default='exp', help='Experiment name')
    
    args = parser.parse_args()
    
    # Import required modules
    import os
    import sys
    import platform
    
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
    
    # Get YOLO version
    version = get_yolo_version()
    print(f"Selected YOLOv{version[1:]} with version code: {version}")
    
    # Double check the version is correct
    if version == "v7" and "YOLOv11" in locals().get('output', ''):
        print("ERROR: Version selection incorrect. Fixing to use YOLOv7...")
        version = "v7"  # Force correct version
    
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
    print(f"Repository directory will be: {repo_dir}")
    
    # Check if repo directory exists
    if not os.path.exists(repo_dir):
        print(f"Cloning YOLOv{version[1:]} repository...")
        try:
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
        except Exception as e:
            print(f"Error cloning repository: {e}")
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
        
        # Install pynvml for better GPU monitoring
        if has_gpu:
            try:
                import pynvml
            except ImportError:
                print("Installing pynvml for better GPU monitoring...")
                run_command(f"{sys.executable} -m pip install pynvml", verbose=False)
    
    # Check for CUDA compatibility issues and fix if possible
    if has_gpu:
        import torch
        try:
            cuda_version_torch = torch.version.cuda
            import re
            import subprocess
            
            # Get system CUDA version
            nvidia_smi = subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True)
            match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi.stdout)
            if match:
                cuda_version_system = match.group(1)
                cuda_major_torch = cuda_version_torch.split('.')[0]
                cuda_major_system = cuda_version_system.split('.')[0]
                
                if cuda_major_torch != cuda_major_system:
                    print(f"\n⚠️ CUDA VERSION MISMATCH DETECTED ⚠️")
                    print(f"PyTorch CUDA: {cuda_version_torch} vs System CUDA: {cuda_version_system}")
                    
                    # Suggest fixing by installing correct PyTorch version
                    print("\nWould you like to install a compatible PyTorch version? (Recommended)")
                    fix_cuda = input("Install compatible PyTorch version? (y/n): ").strip().lower()
                    
                    if fix_cuda.startswith('y'):
                        # Map CUDA system version to PyTorch install command
                        cuda_commands = {
                            "12": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                            "11": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
                            "10": "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu102"
                        }
                        
                        install_cmd = cuda_commands.get(cuda_major_system, cuda_commands["11"])  # Default to CUDA 11.x
                        print(f"\nInstalling PyTorch with compatible CUDA version...")
                        print(f"Running: {install_cmd}")
                        
                        success = run_command(install_cmd, verbose=True)
                        if success:
                            print("PyTorch successfully reinstalled with compatible CUDA version.")
                            # Reimport torch to get new CUDA version
                            import importlib
                            importlib.reload(torch)
                            print(f"New PyTorch CUDA version: {torch.version.cuda}")
                        else:
                            print("Failed to install compatible PyTorch version.")
        except Exception as e:
            print(f"Error checking CUDA compatibility: {e}")
    
    # Find data.yaml file
    yaml_file = next((str(f) for f in Path(data_dir).glob("*.yaml")), None)
    if not yaml_file:
        yaml_file = os.path.join(data_dir, "data.yaml")
        print(f"No YAML config found. Please create {yaml_file} manually.")
        return
    print(f"Using dataset config: {yaml_file}")
    
    # Verify and fix dataset paths in YAML file
    if not verify_dataset_yaml(yaml_file, version):
        print("Failed to verify dataset configuration. Please check the data.yaml file manually.")
        return
    
    # Determine weights file
    weights = args.weights
    if not weights:
        # Default weights based on version and size
        if version == "v5":
            weights = f"yolov5{model_size}.pt"
        elif version == "v6":
            # YOLOv6 naming convention
            weights = f"yolov6_{model_size}.pt"
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
            
    # Check if weights file exists and download if needed
    if weights and not os.path.exists(weights):
        print(f"\nWeights file '{weights}' not found locally.")
        
        # For all YOLO versions, try to download directly from source URL
        if version in YOLO_WEIGHTS and model_size in YOLO_WEIGHTS[version]:
            try:
                import urllib.request
                
                # Get direct download URL
                download_url = YOLO_WEIGHTS[version][model_size]
                print(f"Downloading {weights} directly from: {download_url}")
                
                # Download with progress tracking
                def report_progress(block_num, block_size, total_size):
                    read_so_far = block_num * block_size
                    if total_size > 0:
                        percent = read_so_far * 100 / total_size
                        s = f"\rDownloading {weights}: {percent:.1f}% [{read_so_far} / {total_size}]"
                        sys.stdout.write(s)
                        sys.stdout.flush()
                
                # Perform the download
                urllib.request.urlretrieve(download_url, weights, reporthook=report_progress)
                print(f"\nSuccessfully downloaded {weights}")
                
            except Exception as e:
                print(f"Direct download failed: {e}")
                print("Trying alternative download method...")
                
                # For YOLOv8, v10, v11 use the ultralytics API as fallback
                if version in ["v8", "v10", "v11"]:
                    try:
                        print(f"Attempting to download {weights} via ultralytics...")
                        # Install ultralytics if not already installed
                        run_command(f"{sys.executable} -m pip install ultralytics", verbose=False)
                        
                        # For YOLOv10, YOLOv11 - use YOLOv8 weights since they're compatible
                        if version in ["v10", "v11"]:
                            print(f"YOLOv{version[1:]} uses the same architecture as YOLOv8.")
                            v8_weights = weights.replace(f"v{version[1:]}", "v8")
                            
                            # Download YOLOv8 weights
                            import_cmd = f"from ultralytics import YOLO; model = YOLO('{v8_weights}')"
                            run_command(f"{sys.executable} -c \"{import_cmd}\"", verbose=True)
                            
                            # Rename the file to match the requested version
                            if os.path.exists(v8_weights):
                                import shutil
                                shutil.copy2(v8_weights, weights)
                                print(f"Using YOLOv8 weights for YOLOv{version[1:]} (copied to {weights})")
                            else:
                                print(f"Failed to download {v8_weights}. Will try to download {weights} directly.")
                        
                        # Direct download for YOLOv8
                        import_cmd = f"from ultralytics import YOLO; model = YOLO('{weights}')"
                        run_command(f"{sys.executable} -c \"{import_cmd}\"", verbose=True)
                    except Exception as e:
                        print(f"Ultralytics download failed: {e}")
        else:
            print(f"No direct download URL found for YOLOv{version[1:]} {model_size.upper()}.")
            print("Trying alternative download method...")
            
            # For YOLOv8 or YOLOv11, try using Ultralytics API
            if version in ["v8", "v10", "v11"]:
                try:
                    # Install ultralytics
                    run_command(f"{sys.executable} -m pip install ultralytics", verbose=False)
                    
                    # Try downloading directly via YOLO
                    import_cmd = f"from ultralytics import YOLO; model = YOLO('{weights}')"
                    run_command(f"{sys.executable} -c \"{import_cmd}\"", verbose=True)
                except Exception as e:
                    print(f"Error downloading weights: {e}")
        
        # Check if download was successful
        if os.path.exists(weights):
            print(f"Successfully downloaded {weights}")
        else:
            print(f"Warning: Failed to download {weights}. Training may fail.")
            print("Trying to find a suitable fallback...")
            
            # Try to find a fallback weight file
            if version in ["v8", "v10", "v11"]:
                fallback_weights = "yolov8n.pt"  # Use nano as fallback
                try:
                    # Try to download fallback weights
                    print(f"Attempting to download {fallback_weights} as fallback...")
                    import_cmd = f"from ultralytics import YOLO; model = YOLO('{fallback_weights}')"
                    run_command(f"{sys.executable} -c \"{import_cmd}\"", verbose=True)
                    
                    if os.path.exists(fallback_weights):
                        print(f"Using fallback weights: {fallback_weights}")
                        weights = fallback_weights
                except Exception as e:
                    print(f"Failed to download fallback weights: {e}")
    
    # Prepare model files needed for AMP checks to prevent unnecessary downloads
    if version in ["v8", "v10", "v11"]:
        prepare_amp_check_models(version, model_size)
    
    # Set project and name for the training run
    project_dir = args.project
    exp_name = args.name
    
    print(f"\nResults will be saved to: {os.path.join(project_dir, exp_name)}")
    
    # Create and run training command
    print(f"\nPreparing to train YOLOv{version[1:]} {model_size.upper()} model...")
    
    # Quote paths to handle spaces in Windows
    yaml_file_quoted = f'"{yaml_file}"'
    
    # Set device parameter
    device_param = ""
    if args.device:
        device_param = f"device={args.device}"
    elif has_gpu:
        # For CUDA version mismatch, add a warning but still try to use GPU
        import torch
        cuda_version_torch = torch.version.cuda if hasattr(torch, 'version') and hasattr(torch.version, 'cuda') else "unknown"
        
        # Get system CUDA version from nvidia-smi
        try:
            nvidia_smi = subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True)
            match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi.stdout)
            if match:
                cuda_version_system = match.group(1)
                print(f"\nDetected CUDA versions - PyTorch: {cuda_version_torch}, Driver: {cuda_version_system}")
                print("If you experience GPU utilization issues, this version difference might be the cause.")
        except Exception:
            pass
        
        print(f"\nForcing GPU usage with PyTorch CUDA {cuda_version_torch}")
        device_param = "device=0"  # Use first GPU by default
        
        # For Ultralytics YOLOv8/v10/v11, explicitly force CUDA usage
        if version in ["v8", "v10", "v11"]:
            # Set environment variables to force CUDA usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['PYTHONMEM_LOGS'] = '1'  # Memory debug info
            
            # Set environment variable to control AMP model downloads
            os.environ['ULTRALYTICS_AMP_SKIP_MODEL_CHECK'] = '1'
            
            # Force initialize CUDA device
            if torch.cuda.is_available():
                _ = torch.zeros(1).cuda()
                print(f"GPU initialized: {torch.cuda.get_device_name(0)}")
                
            print("Setting CUDA_VISIBLE_DEVICES=0 to enforce GPU usage")
    else:
        device_param = "device=cpu"
        print("Using CPU for training (will be slow)")
    
    # Set additional parameters for better GPU usage
    if has_gpu:
        # Add cache parameter to speed up training
        cache_param = "cache=True"
        # Reduce image augmentation to speed up processing
        mosaic_param = "mosaic=1"
        # Add mixed precision training for faster GPU computation
        mixed_precision_param = "half=True"
        
        # For YOLOv8/v10/v11, explicitly set amp flag to ensure GPU usage
        if version in ["v8", "v10", "v11"]:
            mixed_precision_param = "amp=True"
            print("Enabling Automatic Mixed Precision (AMP) to ensure GPU usage")
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
        
        # Special case for YOLOv11
        if version == "v11":
            # Verify the weights exist, or use fallback to YOLOv8
            if not os.path.exists(weights):
                fallback_weights = weights.replace("v11", "v8")
                if os.path.exists(fallback_weights):
                    print(f"YOLOv11 weights not found, using YOLOv8 weights: {fallback_weights}")
                    weights = fallback_weights
                elif os.path.exists("yolov8n.pt"):
                    print(f"Using yolov8n.pt as fallback weights")
                    weights = "yolov8n.pt"
                else:
                    print("No suitable weights found. Training may fail.")
            
            # Create a copy of YOLOv8 weights renamed as YOLOv11 to avoid training issues
            if os.path.exists(weights) and "v8" in weights and "v11" not in weights:
                v11_weights = weights.replace("v8", "v11")
                if not os.path.exists(v11_weights):
                    import shutil
                    print(f"Creating YOLOv11 weights file {v11_weights} from {weights}")
                    shutil.copy2(weights, v11_weights)
                    weights = v11_weights
        
        # For v8/v10/v11, use ultralytics API
        cmd = (f"yolo train "
               f"model={weights} data={yaml_file_quoted} epochs={epochs} "
               f"imgsz={args.img_size} batch={batch_size} {device_param} "
               f"workers={0 if platform.system() == 'Windows' else 8} {verbose_param} "
               f"{cache_param} {mosaic_param} {mixed_precision_param} "
               f"project={project_dir} name={exp_name}")
    elif version == "v6":
        # YOLOv6 has a different training script structure
        train_script = os.path.join(repo_dir, "tools", "train.py")
        if not os.path.exists(train_script):
            print(f"Cannot find training script for YOLOv{version[1:]}")
            return
            
        # YOLOv6 specific parameters
        device_arg = f"--device {args.device}" if args.device else f"--device 0" if has_gpu else "--device cpu"
        workers_arg = "--workers 0" if platform.system() == 'Windows' else "--workers 8"
        
        cmd = (f"{sys.executable} {train_script} "
               f"--batch-size {batch_size} --img-size {args.img_size} "
               f"--conf-file {os.path.join(repo_dir, 'configs', f'yolov6_{model_size}.py')} " 
               f"--data-path {yaml_file_quoted} --epochs {epochs} "
               f"{device_arg} {workers_arg} "
               f"--output-dir {project_dir}/{exp_name}")
        
        if weights:
            cmd += f" --weights {weights}"
    else:
        # Traditional YOLO training for v5, v7, v9
        train_script = os.path.join(repo_dir, "train.py")
        if not os.path.exists(train_script):
            train_script = os.path.join(repo_dir, "tools", "train.py")
            if not os.path.exists(train_script):
                print(f"Cannot find training script for YOLOv{version[1:]}")
                return
        
        device_arg = f"--device {args.device}" if args.device else f"--device 0" if has_gpu else "--device cpu"
        workers_arg = "--workers 0" if platform.system() == 'Windows' else "--workers 8"
        
        # Add GPU-specific parameters for YOLOv5/7/9
        gpu_specific = ""
        if has_gpu:
            if version == "v5":
                gpu_specific = "--cache"  # YOLOv5 doesn't support --half flag
            else:
                gpu_specific = "--cache --half"  # Other versions support both flags
        
        # Handle YOLOv5 specific parameters
        if version == "v5":
            # YOLOv5 uses --data for the yaml file
            data_arg = f"--data {yaml_file_quoted}"
            img_size_arg = f"--img {args.img_size}"
            batch_arg = f"--batch-size {batch_size}"
        else:
            # Other versions might use different format
            data_arg = f"--data {yaml_file_quoted}"
            img_size_arg = f"--img-size {args.img_size}"
            batch_arg = f"--batch-size {batch_size}"
        
        cmd = (f"{sys.executable} {train_script} "
               f"{data_arg} --epochs {epochs} "
               f"{batch_arg} {img_size_arg} "
               f"{device_arg} {workers_arg} {gpu_specific} "
               f"--project {project_dir} --name {exp_name}")
        
        if weights:
            cmd += f" --weights {weights}"
        
        if args.verbose or not has_gpu:
            cmd += " --verbose"
    
    # Ask for confirmation
    print("\nTraining command:")
    print(cmd)
    
    # Extract model weights from command
    weights_in_cmd = ""
    for part in cmd.split():
        if part.startswith("model="):
            weights_in_cmd = part.split("=")[1]
            break
    
    # Verify model weights exist before training
    if weights_in_cmd and not os.path.exists(weights_in_cmd):
        print(f"\n⚠️ WARNING: Model weights '{weights_in_cmd}' not found!")
        
        # Check if we can find a valid alternative
        alternative_weights = []
        
        # Look for version-specific alternatives
        if "v5" in weights_in_cmd:
            for size in ["n", "s", "m", "l", "x"]:
                alt = f"yolov5{size}.pt"
                if os.path.exists(alt):
                    alternative_weights.append(alt)
        
        elif "v6" in weights_in_cmd:
            for size in ["n", "s", "m", "l"]:
                alt = f"yolov6_{size}.pt"
                if os.path.exists(alt):
                    alternative_weights.append(alt)
        
        elif "v7" in weights_in_cmd:
            if os.path.exists("yolov7.pt"):
                alternative_weights.append("yolov7.pt")
            if os.path.exists("yolov7-tiny.pt"):
                alternative_weights.append("yolov7-tiny.pt")
        
        elif "v8" in weights_in_cmd or "v10" in weights_in_cmd or "v11" in weights_in_cmd:
            # Check for any YOLOv8 weights
            for size in ["n", "s", "m", "l", "x"]:
                alt = f"yolov8{size}.pt"
                if os.path.exists(alt):
                    alternative_weights.append(alt)
        
        elif "v9" in weights_in_cmd:
            if os.path.exists("yolov9-c.pt"):
                alternative_weights.append("yolov9-c.pt")
            if os.path.exists("yolov9-e.pt"):
                alternative_weights.append("yolov9-e.pt")
        
        # Suggest alternatives if found
        if alternative_weights:
            print("Found alternative weight files:")
            for i, alt in enumerate(alternative_weights, 1):
                print(f"{i}. {alt}")
            
            choice = input(f"Select an alternative (1-{len(alternative_weights)}) or press Enter to try downloading: ")
            
            if choice.isdigit() and 1 <= int(choice) <= len(alternative_weights):
                selected_alt = alternative_weights[int(choice) - 1]
                cmd = cmd.replace(weights_in_cmd, selected_alt)
                print(f"Using alternative weights: {selected_alt}")
                print("\nUpdated training command:")
                print(cmd)
            else:
                # Try direct download
                print(f"Attempting to download {weights_in_cmd}...")
                
                try:
                    # For YOLOv8/10/11, try using ultralytics
                    if any(v in weights_in_cmd for v in ["v8", "v10", "v11"]):
                        import_cmd = f"from ultralytics import YOLO; model = YOLO('{weights_in_cmd}')"
                        run_command(f"{sys.executable} -c \"{import_cmd}\"", verbose=True)
                    
                    # For YOLOv8, try to use it as a fallback for other versions
                    elif not os.path.exists("yolov8n.pt"):
                        print("Downloading yolov8n.pt as fallback...")
                        import_cmd = f"from ultralytics import YOLO; model = YOLO('yolov8n.pt')"
                        run_command(f"{sys.executable} -c \"{import_cmd}\"", verbose=True)
                        
                        if os.path.exists("yolov8n.pt"):
                            if input("Use yolov8n.pt as fallback? (y/n): ").lower().startswith('y'):
                                cmd = cmd.replace(weights_in_cmd, "yolov8n.pt")
                                print("\nUpdated training command:")
                                print(cmd)
                except Exception as e:
                    print(f"Error downloading weights: {e}")
    
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
                
                # Verify if GPU was actually used
                if has_gpu:
                    import torch
                    if torch.cuda.is_available():
                        # Get peak memory usage - this will only be non-zero if GPU was used
                        peak_memory = torch.cuda.max_memory_allocated(0) / (1024**3)  # GB
                        
                        if peak_memory > 0.1:  # More than 100MB was used
                            print(f"\nGPU was successfully used! Peak memory usage: {peak_memory:.2f} GB")
                        else:
                            print("\nWARNING: GPU was detected but may not have been used effectively during training.")
                            print("Possible reasons:")
                            print("1. CUDA version mismatch between PyTorch and system")
                            print("2. Training script did not properly utilize GPU")
                            print("3. Dataset was too small to benefit from GPU acceleration")
                            print("\nTo fix CUDA version mismatch:")
                            print(f"Your system has CUDA {torch.version.cuda}")
                            print("Install matching PyTorch version from: https://pytorch.org/get-started/locally/")
                            print("For example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
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