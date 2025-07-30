# Introduction

This directory contains a simulated implementation of a privacy attack on a benchmark for facial recognition algorithms described in our KDD 2025 paper [Benchmarking Fraud Detectors on Private Graph Data](https://dl.acm.org/doi/10.1145/3690624.3709170), by Alexander Goldberg, Giulia Fanti, Nihar Shah, and Zhiwei Steven Wu. 

For a technical summary of the simulated attack on the NIST benchmark and our results, see [attack_writeup.pdf](https://github.com/akgoldberg/face_recognition_privacy_attack/blob/master/attack_writeup.pdf).

# Main Files

- `run.py`: This is the main python script to execute the face recognition privacy attack. 
- `run_attack.ipynb`: A Jupyter notebook that provides an interactive way to run the face recognition privacy attack. 
- `benchmark_1N.py`: Code to simulate a 1:N facial recognition benchmark.
- `attack.py`: The core logic of the privacy attack on the 1:N benchmark.
- `load_data.py`: Code to load CelebA dataset.

# Directory Setup 

We use [docker](https://www.docker.com/) to install all necessary requirements.

1. Setup docker container:

    *If running from command line:* 
    
    Build the docker image (from directory .devcontainer):  
    > docker build -t fr_benchmark .

    Run the docker container:
    > docker run -it -p 8888:8888 --rm -v "$(pwd):"/app fr_benchmark

    *If using VS Code:*
    
    Use Dev Container extension to open in remote container.

    To run jupyter notebok, once docker container running, execute:
        > jupyter notebook --ip=0.0.0.0 --allow-root


2. Run attack either with python script: 
    > python run.py

    or using Jupyter notebook run_attack.ipynb.
