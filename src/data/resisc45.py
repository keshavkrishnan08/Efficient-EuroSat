"""
NWPU-RESISC45 dataset helper.

Downloads and extracts the RESISC45 remote sensing scene classification
dataset (Cheng et al., 2017) if not already present.  The dataset
contains 31,500 images across 45 scene classes (700 images per class),
with images sized 256x256.

Since RESISC45 is not available through torchvision, we provide a
download helper that fetches the dataset and sets it up for use with
torchvision.datasets.ImageFolder.
"""

import os
import zipfile
import logging

logger = logging.getLogger(__name__)

# OneDrive direct download link for RESISC45
# Source: https://gcheng-nwpu.github.io/#Datasets
RESISC45_URL = (
    "https://storage.googleapis.com/remote_sensing_representations/"
    "resisc45.zip"
)

RESISC45_DIR_NAME = "NWPU-RESISC45"


def ensure_resisc45(root: str = "./data") -> str:
    """Ensure RESISC45 dataset exists at root, download if needed.

    Parameters
    ----------
    root : str
        Parent directory for datasets.

    Returns
    -------
    str
        Path to the ImageFolder-compatible directory containing
        class subdirectories.
    """
    resisc_root = os.path.join(root, RESISC45_DIR_NAME)

    if os.path.isdir(resisc_root) and _has_class_dirs(resisc_root):
        logger.info("RESISC45 found at %s", resisc_root)
        return resisc_root

    print(f"  Downloading RESISC45 dataset to {root}...")
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "resisc45.zip")

    _download_file(RESISC45_URL, zip_path)

    print("  Extracting RESISC45...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    # Clean up zip
    os.remove(zip_path)

    if not os.path.isdir(resisc_root):
        # Handle alternative extraction names
        for name in os.listdir(root):
            candidate = os.path.join(root, name)
            if os.path.isdir(candidate) and _has_class_dirs(candidate):
                os.rename(candidate, resisc_root)
                break

    if not os.path.isdir(resisc_root) or not _has_class_dirs(resisc_root):
        raise RuntimeError(
            f"RESISC45 extraction failed. Expected class directories at {resisc_root}. "
            "Please download manually from https://gcheng-nwpu.github.io/#Datasets "
            f"and extract to {root}/NWPU-RESISC45/"
        )

    print(f"  RESISC45 ready: {_count_classes(resisc_root)} classes at {resisc_root}")
    return resisc_root


def _download_file(url: str, dest: str):
    """Download a file with progress reporting."""
    import urllib.request
    import shutil

    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB
            while True:
                data = response.read(block_size)
                if not data:
                    break
                out_file.write(data)
                downloaded += len(data)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r  Downloaded: {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
            print()
    except Exception as e:
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            f"Failed to download RESISC45 from {url}: {e}\n"
            "Please download manually from https://gcheng-nwpu.github.io/#Datasets "
            f"and extract to the data directory."
        ) from e


def _has_class_dirs(path: str) -> bool:
    """Check if a directory contains at least 10 subdirectories (class folders)."""
    if not os.path.isdir(path):
        return False
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return len(subdirs) >= 10


def _count_classes(path: str) -> int:
    """Count the number of class subdirectories."""
    return len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
