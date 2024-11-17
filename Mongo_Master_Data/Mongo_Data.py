import requests
import json
from tqdm import tqdm
import os

# Replace with your public S3 URL
url = 'https://mongojsondumptemp.s3.ap-south-1.amazonaws.com/product_service_production/variant_image_vectors.json'

# Target directory where you want to save the files
target_directory = r"D:\Vectors_Data_2"

# Define how many records each file should contain
records_per_file = 100000

# Define the number of files to create
num_files = 50

def download_file_in_chunks(url, target_directory, records_per_file, num_files):
    try:
        # Create the target directory if it doesn't exist
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # Initialize counters and storage
        record_count = 0
        file_count = 1
        current_records = []
        
        # Track if download needs to resume
        file_path = os.path.join(target_directory, f'variants_image_vectors_{file_count}.json')
        downloaded_records = 0

        # Check if any files already exist
        if os.path.exists(file_path):
            # Resume from where it left off
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        current_records.append(data)
                    except json.JSONDecodeError:
                        continue
            
            print(f"Resuming from {record_count} records in {file_path}")

        # Download data in chunks
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Use tqdm for progress tracking
        with tqdm(total=None, unit='B', unit_scale=True, desc=f"Downloading", ascii=True) as pbar:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    pbar.update(len(line))

                    # Parse the JSON line
                    try:
                        data = json.loads(line)
                        current_records.append(data)
                        record_count += 1
                    except json.JSONDecodeError:
                        continue

                    # Save to file if we reach the required number of records
                    if record_count >= records_per_file:
                        with open(file_path, 'a', encoding='utf-8') as f:
                            json.dump(current_records, f, indent=4)
                            f.write("\n")
                        print(f"Saved {record_count} records to {file_path}")

                        # Reset counters for the next file
                        current_records = []
                        record_count = 0
                        file_count += 1

                        # Exit if max number of files is reached
                        if file_count > num_files:
                            break
                        # Update file path for the next file
                        file_path = os.path.join(target_directory, f'variants_image_vectors_{file_count}.json')

            # Save any remaining records
            if current_records and file_count <= num_files:
                with open(file_path, 'a', encoding='utf-8') as f:
                    json.dump(current_records, f, indent=4)
                    f.write("\n")
                print(f"Saved remaining {len(current_records)} records to {file_path}")

        print("\nDownload and file saving completed successfully.")
    
    except requests.ConnectionError:
        print("Connection error. Please check your internet connection and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function
download_file_in_chunks(url, target_directory, records_per_file, num_files)
