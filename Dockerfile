# MADDPG Adversarial Robustness — Docker Setup
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /workspace

# System dependencies — nvidia-utils removed (already in base image)
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional packages
RUN pip install --no-cache-dir \
    torch-geometric \
    jupyter \
    ipython \
    tensorboard \
    wandb \
    GPUtil \
    psutil

# Source code
COPY src/ ./src/
COPY src/standalone_experiment_runner.py ./src/standalone_experiment_runner.py
COPY tools/ ./tools/
COPY experiment_config.json .
COPY pyproject.toml .

# Install package in editable mode
RUN pip install -e .

# Directories that map to Docker volume mounts
RUN mkdir -p /workspace/data/results \
             /workspace/data/models \
             /workspace/logs

# Python path — covers both sub-packages the runner imports from
ENV PYTHONPATH=/workspace/src:/workspace/src/maddpg_clean:/workspace/src/attack_framework
ENV PYTHONUNBUFFERED=1

# Default: run full experiment (all three phases)
CMD ["python", "src/standalone_experiment_runner.py", \
     "--config", "experiment_config.json", \
     "--gpu", "0", \
     "--phase", "all", \
     "--results-dir", "data/results/main_run"]