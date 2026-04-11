#!/usr/bin/env python3
import argparse

from macrophage_analysis.io import sort_microscopy_images

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
