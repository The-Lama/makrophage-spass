# Makrophage Notebook Setup

This repository packages a JupyterLab environment for the image-analysis notebook in this folder. The Docker image already contains TensorFlow plus the Python packages from `requirements.txt`:

- `jupyterlab`
- `stardist`
- `scikit-image`
- `matplotlib`
- `tifffile`

The recommended way to run this project is Docker. A plain `pip install -r requirements.txt` is not a complete local setup, because TensorFlow comes from the base Docker image.

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

Open that link in the browser, then open `Spassprojekt.ipynb` in JupyterLab.

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