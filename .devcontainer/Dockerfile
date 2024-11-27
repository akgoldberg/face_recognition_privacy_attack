# Dockerfile
FROM python:3.9-slim

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git wget unzip build-essential libgl1-mesa-glx

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Expose Jupyter port
EXPOSE 8888

# # Start a bash shell on container start
# CMD ["/bin/bash"] 
# Start Jupyter notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]