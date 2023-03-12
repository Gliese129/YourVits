#!/bin/bash
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N job_name

# Load CUDA and Python
module load cuda/11.7
module load python/3.10.9

# Create a virtual environment
python -m venv env
source env/bin/activate

# download program
git clone https://github.com/Gliese129/YourVits.git

# shellcheck disable=SC2164
cd YourVits


# Install requirements
pip install -r requirements.txt -y
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 -y

# Run Python code
python your_script.py

# Deactivate environment
deactivate
