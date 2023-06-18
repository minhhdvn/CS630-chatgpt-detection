# CS630-chatgpt-detection

This is the repository for the project: "X-As-A-Service: Building a Cloud Service for Detecting ChatGPT Generated Content".

Student: Minh Nguyen

Please follow the following steps to run the cloud service.

1. First, please create a conda environment and install the following packages:
```
conda create -n cs630 python=3.6
source activate cs630

pip install transformers==3.5.1
pip install sentencepiece==0.1.91
pip install torch==1.7.1
pip install web.py==0.62
pip install protobuf==3.20.0
```
2. Next, we need to a checkpoint for the trained model, which can be downloaded at: https://drive.google.com/file/d/1cz8ylvOZ5QyWWeMUv1ZCueGQ8FytyiJy/view?usp=sharing . The checkpoint should be placed under `./logs/`.
3. To start the server, please use the command: `python run_server.py`
4. To test the service, please use the command: `python run_client.py`
