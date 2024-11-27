# Setup 

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


2. 