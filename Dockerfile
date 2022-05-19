FROM walkerlab/pytorch:python3.8-torch1.10.0-cuda11.3.1-dj0.12.7

# copy this project and install
COPY . /src/FENS-2022
RUN pip install -e /src/FENS-2022

# Disable password system for Jupyter Notebook
ENV JUPYTER_PASSWORD=''

WORKDIR /content

