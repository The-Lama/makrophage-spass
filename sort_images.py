#!/usr/bin/env python3
import shutil
import argparse
from pathlib import Path

def sort_microscopy_images(source_dir, dest_dir):
    """
    Scans the source directory for .tif files and sorts them into a 
    Donor/Condition/ nested folder structure inside the destination directory.
    """
    source_path = Path(source_dir).resolve()
    dest_path = Path(dest_dir).resolve()
    
    print(f"Scanning source directory: {source_path}")
    
    # Check if the source actually exists
    if not source_path.exists():
        print(f"Error: The source directory '{source_path}' does not exist.")
        return

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .tif and .tiff files in the source directory
    image_files = list(source_path.glob("*.tif")) + list(source_path.glob("*.tiff"))
    
    if not image_files:
        print("No loose .tif files found to sort in the source directory.")
        return

    moved_count = 0
    skipped_count = 0

    for file_path in image_files:
        file_name = file_path.name
        parts = file_name.split("_")
        
        # Safety check: ensure the file matches our expected naming convention
        if len(parts) < 3:
            print(f"Skipping '{file_name}': Doesn't match expected Donor_Condition_ format.")
            skipped_count += 1
            continue
            
        donor_id = parts[0]
        condition = parts[1]
        
        # Build the destination path inside the target directory
        final_dest_dir = dest_path / donor_id / condition
        final_dest_dir.mkdir(parents=True, exist_ok=True)
        
        destination_file = final_dest_dir / file_name
        
        # Move the file
        shutil.move(str(file_path), str(destination_file))
        moved_count += 1

    print("-" * 30)
    print("Sorting Complete!")
    print(f"Successfully moved: {moved_count} files.")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} unrecognized files.")
    print(f"Data is securely stored in: {dest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort macrophage microscopy images into a nested structure.")
    parser.add_argument(
        "-s", "--source", 
        type=str, 
        required=True,
        help="The folder containing the loose .tif files to unpack."
    )
    parser.add_argument(
        "-d", "--dest", 
        type=str, 
        default="./raw_data", 
        help="The folder to save the sorted data into (defaults to './raw_data')."
    )
    
    args = parser.parse_args()
    sort_microscopy_images(args.source, args.dest)