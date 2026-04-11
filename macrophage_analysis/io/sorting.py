from __future__ import annotations

import shutil
from pathlib import Path

from .catalog import clear_image_catalog_cache


def parse_donor_condition_from_filename(filename: str) -> tuple[str, str] | None:
    parts = filename.split("_", 2)
    if len(parts) < 3:
        return None
    donor_id, condition = parts[0].strip(), parts[1].strip()
    if not donor_id or not condition:
        return None
    return donor_id, condition


def sort_microscopy_images(source_dir: str | Path, dest_dir: str | Path) -> None:
    """
    Scan a source directory for loose TIFF files and move them into
    Donor/Condition/ nested folders inside the destination directory.
    """
    source_path = Path(source_dir).resolve()
    dest_path = Path(dest_dir).resolve()

    print(f"Scanning source directory: {source_path}")

    if not source_path.exists():
        print(f"Error: The source directory '{source_path}' does not exist.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)
    image_files = sorted(source_path.glob("*.tif")) + sorted(source_path.glob("*.tiff"))

    if not image_files:
        print("No loose .tif files found to sort in the source directory.")
        return

    moved_count = 0
    skipped_count = 0

    for file_path in image_files:
        donor_condition = parse_donor_condition_from_filename(file_path.name)
        if donor_condition is None:
            print(
                f"Skipping '{file_path.name}': Doesn't match expected Donor_Condition_ format."
            )
            skipped_count += 1
            continue

        donor_id, condition = donor_condition
        final_dest_dir = dest_path / donor_id / condition
        final_dest_dir.mkdir(parents=True, exist_ok=True)

        destination_file = final_dest_dir / file_path.name
        shutil.move(str(file_path), str(destination_file))
        moved_count += 1

    print("-" * 30)
    print("Sorting Complete!")
    print(f"Successfully moved: {moved_count} files.")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} unrecognized files.")
    print(f"Data is securely stored in: {dest_path}")
    clear_image_catalog_cache()
