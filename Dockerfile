# Use Google's official, lightweight CPU-only TensorFlow image
FROM tensorflow/tensorflow:2.15.0

# Install system-level C++ and graphics libraries needed by scikit-image
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set our working directory inside the container
WORKDIR /app

# Copy our requirements file and install the Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter's default port
EXPOSE 8888

# Command to launch JupyterLab when the container starts
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]