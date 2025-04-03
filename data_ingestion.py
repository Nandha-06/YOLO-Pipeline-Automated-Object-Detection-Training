import os
import gdown
import argparse
import re
import zipfile
import shutil
import json
import tempfile

def extract_file_id(url):
    """Extract file ID from various Google Drive URL formats"""
    file_id = None
    
    # Handle URLs like 'https://drive.google.com/file/d/{file_id}/view?usp=share_link'
    file_pattern = r'(?:drive\.google\.com/file/d/|drive\.google\.com/open\?id=|drive\.google\.com/uc\?id=)([^/&?]+)'
    match = re.search(file_pattern, url)
    
    if match:
        file_id = match.group(1)
    
    return file_id

def unzip_file(zip_path, extract_dir=None, delete_zip=True):
    """
    Extract contents from a zip file
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract files to. If None, extracts to a folder with the same name as the zip file
        delete_zip: Whether to delete the zip file after extraction
        
    Returns:
        Path to the extracted directory
    """
    try:
        if not os.path.exists(zip_path):
            print(f"Zip file not found: {zip_path}")
            return None
            
        # If extract directory not specified, create one with the same name as the zip file (without extension)
        if extract_dir is None:
            extract_dir = os.path.splitext(zip_path)[0]
            
        # Create extraction directory if it doesn't exist
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            
        # Extract the files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        print(f"Extracted zip contents to: {extract_dir}")
        
        # Delete the zip file if requested
        if delete_zip:
            try:
                os.remove(zip_path)
                print(f"Deleted zip file: {zip_path}")
            except Exception as e:
                print(f"Warning: Could not delete zip file: {str(e)}")
        
        return extract_dir
        
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return None
    except Exception as e:
        print(f"Error extracting zip file: {str(e)}")
        return None

def download_from_drive(url, output_path=None, is_folder=False, quiet=False, extract_zip=True, delete_zip=True):
    """Downloads a file or folder from Google Drive using gdown"""
    try:
        if output_path and not is_folder:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        elif output_path and is_folder:
            os.makedirs(output_path, exist_ok=True)
        
        if is_folder:
            return gdown.download_folder(url=url, output=output_path, quiet=quiet)
        else:
            # Try to extract file ID for better compatibility
            file_id = extract_file_id(url)
            if file_id:
                url = f"https://drive.google.com/uc?id={file_id}"
            
            # Use fuzzy option to handle various URL formats
            downloaded_path = gdown.download(url=url, output=output_path, quiet=quiet, fuzzy=True)
            
            # Check if the downloaded file is a zip and extract it if needed
            if downloaded_path and extract_zip and downloaded_path.lower().endswith('.zip'):
                extract_dir = unzip_file(downloaded_path, delete_zip=delete_zip)
                
                # Save the extraction path to a file for data_validation.py to use
                if extract_dir:
                    save_extraction_path(extract_dir)
                    
                return extract_dir or downloaded_path
            
            return downloaded_path
    except Exception as e:
        print(f"Error downloading from {url}: {str(e)}")
        return None

def save_extraction_path(path):
    """Save the extraction path to a file for later use by data_validation.py"""
    data = {
        'last_extraction_path': os.path.abspath(path),
        'timestamp': import_time()
    }
    
    # Save to a file in the system's temp directory instead of current directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, 'yolo_extraction_path.json')
    
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    # Print message with less visibility
    print(f"Extraction path stored for validation")

def import_time():
    """Import time module and return current time"""
    import time
    return time.time()

def get_user_input():
    """Get URL from user via interactive input"""
    url = input("Enter Google Drive URL: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        return None, None, False, False, True, False, True
    
    return [url], None, False, False, True, False, True

def validate_dataset(data_path, create_yaml=False):
    """Validate dataset after downloading by running data_validation.py"""
    try:
        # Import validate_dataset function from data_validation
        from data_validation import validate_dataset, print_validation_results, create_yaml_template
        
        # Run validation
        print(f"\nValidating dataset at: {data_path}")
        results = validate_dataset(data_path)
        print_validation_results(results)
        
        # Create YAML template if requested and missing
        if create_yaml and not results["yaml_config"]["valid"]:
            create_yaml_template(data_path)
        
        return results["valid"]
    except ImportError:
        print("Warning: data_validation.py not found. Cannot validate dataset.")
        return None
    except Exception as e:
        print(f"Error validating dataset: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download files from Google Drive')
    parser.add_argument('urls', nargs='*', help='Google Drive URLs or file IDs')
    parser.add_argument('-o', '--output', help='Output path for downloaded file(s)')
    parser.add_argument('-f', '--folder', action='store_true', help='Download entire folder')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress download progress')
    parser.add_argument('-n', '--no-extract', action='store_true', help='Do not extract zip files')
    parser.add_argument('-v', '--validate', action='store_true', help='Validate dataset after downloading')
    parser.add_argument('-y', '--create-yaml', action='store_true', help='Create YAML template if missing')
    parser.add_argument('-k', '--keep-zip', action='store_true', help='Keep zip files after extraction')
    
    args = parser.parse_args()
    
    # Get URL via input if no URL provided
    if not args.urls:
        urls, output, is_folder, quiet, extract_zip, validate, delete_zip = get_user_input()
        if not urls:
            return
        create_yaml = False
    else:
        urls = args.urls
        output = args.output
        is_folder = args.folder
        quiet = args.quiet
        extract_zip = not args.no_extract
        validate = args.validate
        create_yaml = args.create_yaml
        delete_zip = not args.keep_zip
    
    last_result = None
    
    for url in urls:
        print(f"Processing: {url}")
        result = download_from_drive(url, output, is_folder, quiet, extract_zip, delete_zip)
        last_result = result
        
        if result:
            if is_folder:
                print(f"Successfully downloaded {len(result)} files from folder")
            else:
                print(f"Successfully downloaded to: {result}")
        else:
            print(f"Failed to download from: {url}")
    
    # Always validate if we have a valid download result (directory)
    if last_result and os.path.isdir(last_result):
        validate_dataset(last_result, create_yaml)

if __name__ == "__main__":
    main() 