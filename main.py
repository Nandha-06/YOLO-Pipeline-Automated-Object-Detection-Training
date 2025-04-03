#!/usr/bin/env python
import os
import sys
import time
from pathlib import Path

def print_step(step_name):
    """Print a formatted step header"""
    print("\n" + "="*60)
    print(f" {step_name} ".center(60, "="))
    print("="*60)

def main():
    # Create a log file for the execution
    log_file = f"yolo_workflow_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Recording execution log to: {log_file}")
    
    try:
        print_step("STEP 1: DATA INGESTION")
        
        # Import data ingestion module
        from data_ingestion import download_from_drive, get_user_input
        
        # Ask user for data source
        print("Please provide a Google Drive URL for your dataset:")
        urls, output, is_folder, quiet, extract_zip, validate, delete_zip = get_user_input()
        
        if not urls:
            print("No data source provided. Exiting workflow.")
            return False
        
        # Download and extract the dataset
        data_path = None
        for url in urls:
            print(f"Downloading from: {url}")
            result = download_from_drive(
                url=url,
                output_path=output,
                is_folder=is_folder,
                quiet=quiet,
                extract_zip=extract_zip,
                delete_zip=delete_zip
            )
            data_path = result
        
        if not data_path or not os.path.exists(data_path):
            print("Data ingestion failed. Cannot proceed to validation.")
            return False
            
        # Wait a moment before proceeding to validation
        time.sleep(1)
        
        print_step("STEP 2: DATA VALIDATION")
        
        # Import validation module
        from data_validation import validate_dataset, print_validation_results, create_yaml_template
        
        # Validate the dataset
        print(f"Validating dataset at: {data_path}")
        validation_results = validate_dataset(data_path)
        print_validation_results(validation_results)
        
        # Create YAML if needed
        if not validation_results["yaml_config"]["valid"]:
            print("\nNo YAML configuration found. Creating a template...")
            create_yaml_template(data_path)
            print("Please review and edit the generated YAML file before proceeding.")
            
            # Ask user if they want to continue
            proceed = input("\nContinue to training? (y/n): ").strip().lower()
            if not proceed.startswith('y'):
                print("Workflow paused at validation step. Please run training separately.")
                return True
        
        # Wait a moment before proceeding to training
        time.sleep(1)
        
        print_step("STEP 3: MODEL TRAINING")
        
        # Import training module
        from model_training import train_model
        
        # Run the training process
        print("Starting model training...")
        train_model()
        
        print_step("WORKFLOW COMPLETE")
        print("The YOLO workflow has been completed successfully.")
        return True
        
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
        return False
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 