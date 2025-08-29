#!/usr/bin/env python3
"""
Script to remove semantic_id fields from SC3K annotation JSON files.
"""

import json
import argparse
from pathlib import Path


def remove_semantic_id_from_json(json_file_path):
    """Remove semantic_id fields from all keypoints in a JSON annotation file."""
    print(f"Processing {json_file_path}")
    
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    
    # Remove semantic_id from all keypoints
    for annotation in annotations:
        for keypoint in annotation['keypoints']:
            if 'semantic_id' in keypoint:
                del keypoint['semantic_id']
    
    # Save the modified JSON
    with open(json_file_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Removed semantic_id fields from {len(annotations)} annotations")


def main():
    parser = argparse.ArgumentParser(description="Remove semantic_id fields from SC3K annotation JSON files")
    parser.add_argument("json_files", nargs="+", help="Paths to JSON annotation files")
    
    args = parser.parse_args()
    
    for json_file in args.json_files:
        if Path(json_file).exists():
            remove_semantic_id_from_json(json_file)
        else:
            print(f"Warning: {json_file} not found")


if __name__ == "__main__":
    main()