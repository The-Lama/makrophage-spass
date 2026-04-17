# Makrophage Notebook Setup

This repository packages a JupyterLab environment for the image-analysis notebooks in `notebooks/`. The Docker image already contains TensorFlow plus the Python packages from `requirements.txt`:

- `jupyterlab`
- `stardist`
- `scikit-image`
- `matplotlib`
- `tifffile`

The recommended way to run this project is Docker. A plain `pip install -r requirements.txt` is not a complete local setup, because TensorFlow comes from the base Docker image.

## Data Preparation (Automated Sorting)

Before running the StarDist analysis, organize raw `.tif` or `.tiff` files into a nested `raw_data/Donor/Condition/` folder structure.

This repository includes a standalone Python utility, `sort_images.py`, for that step. It does not require Docker, TensorFlow, or the notebook environment.

### How to unpack new data

When you receive a folder of loose microscopy images, run the sorting script from the repository root and point `-s` at the folder containing the new files:

```bash
python3 sort_images.py -s "path/to/new_images_folder"
```

By default, sorted files are moved into `./raw_data`. To write them somewhere else, pass `-d`:

```bash
python3 sort_images.py -s "path/to/new_images_folder" -d "/path/to/raw_data"
```

The script reads the first two underscore-separated filename fields as `Donor` and `Condition`. For example, a file named `D45_B68KCP2_example.tif` will be moved into `raw_data/D45/B68KCP2/`.

Expected output structure:

```text
raw_data/
├── D45/
│   ├── B68KCP2/
│   │   ├── D45_B68KCP2_...tif
│   │   └── ...
│   ├── M0/
│   └── ...
└── D47/
    ├── B68KCP2/
    └── ...
```

Files that do not match the expected `Donor_Condition_...` naming pattern are skipped and reported.

## Installation Guide for Nathalie

### 1. Install Docker Desktop

Download Docker Desktop for Mac, install it like a normal Mac app, and start it once. Wait until Docker Desktop shows that the Docker engine is running before continuing.

If macOS asks whether Docker may access the project folder, allow it.

### 2. Get the project folder onto the Mac

```bash
git clone https://github.com/The-Lama/makrophage-spass.git
cd makrophage-spass
```

### 3. Start JupyterLab

In Terminal, go to the project folder and run:

```bash
cd makrophage-spass
docker compose up --build
```

Notes:

- The first start can take a while on an old MacBook because Docker has to download the base image and install the Python packages.
- This uses the CPU-only setup from `Dockerfile` and `docker-compose.yml`.

### 4. Open the notebook in the browser

After startup, the terminal will print a URL that looks like this:

```text
http://127.0.0.1:8888/lab?token=...
```

Open that link in the browser, then open `notebooks/fluorescence_analysis_workflow.ipynb` in JupyterLab.

### 5. Stop the project

When done, return to Terminal and press `Ctrl+C`.

The next time, you only need:

```bash
docker compose up
```

### Useful things to know

- Files are synced between the Mac and the container, because the project folder is mounted into `/app`.
- Anything saved in JupyterLab stays in the project folder on the Mac.
- Ignore `Dockerfile.gpu` and `docker-compose.gpu.yml`.
- If Docker says the engine is not running, open Docker Desktop and wait a minute, then try again.

## Alex Command Reference

With an NVIDIA GPU, your normal command is the GPU version:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

For later starts without rebuilding:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

This GPU command does two things:

- it switches the build to `Dockerfile.gpu`
- it requests an NVIDIA GPU for the container
