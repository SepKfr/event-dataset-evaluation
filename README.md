# Introduction 

Predictive maintenance (PdM) is an intelligent manufacturing solution that is part of the digital transformation of the manufacturing industry. It utilizes machine-collected data to make predictions about future failures. Within Bosch, sensors continuously monitor production lines, measuring various machine parameters such as component positions, angles, applied force, pressure, and more.

Real-time monitoring is achieved through an Anomaly Detection Engine (ADE), which analyzes the sensor readings. The ADE system identifies different types of anomalies, including outliers, sudden changes in mean or variance, gradual mean shifts, and increases in zero values. When an anomaly occurs, its details are recorded in a real-time database. 
While some anomalies may be non-critical and triggered by rule-based flagging, 
others could indicate underlying machine faults that require correction. 
In some cases, these faults can result in machine breakdowns, causing interruptions in the factory line. Whenever an interruption occurs, a separate system records the details of the event. 

The ultimate goal is to predict potential interruptions preceded by a series of anomalies, allowing for preventive measures that save on financial and maintenance costs.

Below is the list of public datasets that our model is trained and evaluted on to show its generalization to various datasets.

Link to download each dataset is given below:

1. [US Accident](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents): A countrywide car accident dataset that covers 49 states of the USA. The accident data were collected from February 2016 to March 2023. This dataset contains a severity attributes, a number between 1 and 4, where 1 indicates the least impact on traffic, whereas 4 indicates the most impact on traffic. Target labels are created by classifying accidents with severity from 1 to 3 as normal (not fatal) and accidents with severity 4 as abnormal (fatal). 

2. [Severe Weather Data Inventory (SWDI)](https://www.kaggle.com/datasets/noaa/severe-weather-data-inventory): An integrated dataset of severe data records of the USA. We utilize the dataset available on Kaggle, containing records from the year 2015 in SWDI. These records are sourced from the National Climatic Data Center archive and encompass a wide array of weather phenomena. This datasets contains the information of size and the probability of severity of each sever event. We classified events based on the probability of severity from 1 to 10, with 1 being the least severe and 10 being the most severe event. Target variables are created by classifying events with severity from 1 to 5 as normal and 5 to 10 as abnormal. 

3. [3W Dataset-Undesirable Events in Oil Wells](https://www.kaggle.com/datasets/afrniomelo/3w-dataset?select=3W): A comprehensive dataset of simulated and hand-drown instances of eight types of undesirable events characterized by eight process variables spanning from year 2012 to 2018. This dataset also contains normal events that are not dangerous or fatal. Undesirable events with types from 1 to 8 are considered as abnormal and events with type 0 are considered as normal. 

# Abstract of Manuscript

Anomaly detection poses a significant challenge, 
primarily due to the problem of class imbalance. 
Traditional solutions like class weight adjustment 
and feature engineering have been explored, 
but we seek to introduce an innovative approach. 
However, we aim to enhance the predictive capacity of the 
classification model through an ensemble technique. 
In this study, we propose a joint residual-classification 
framework that directs the classification model towards identifying 
overarching patterns, while entrusting the residual model 
with capturing intricate particulars. 
Our experimental results demonstrate that our proposed residual-classification 
model is able to outperform a standalone classification model predominately.

# Quickstart

1. Download Miniconda3 using the following link: [Miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh)
2. Set up ate activate Python 3.8 virtual environment using the following commands.

```
conda create -n myenv python=3.8
source activate venv
```

3. Install the requirements 

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
conda install -c conda-forge optuna or conda install -c "conda-forge/label/cf202003" optuna
conda install numpy 
conda install -c anaconda scikit-learn
```

4. Required Versions

```
python >= 3.8.0
torch >= 1.13.0
optuna >= 3.0.4
numpy >= 1.23.5
scikit learn >= 1.2.2
```

# Workflow

1. Download and process each dataset:
After downloading the data, uzip the file and move it to your desired directory. 
Let's say the unzipped data for rare oil events is in my home directory ~/3W/. The following 
command lines process the data and save it in a csv file.
```commandline
command line args:
expt_name: str {oil, sev_weather, us_accident}
python data_loader.py --expt_name [expt_name] --data_path [path_to_data]
One example: 
python data_loader.py --expt_name oil --data_path ~/3w/
```

2. Build and Test

To train and evaluate the deep classification models (including base classification model, 
ablation on adjusting weights, and our residual-classification model) run the below code:

```
command line args:
name:str  name of the deep learning model, e.g. Autoformer
cuda:str Which GPU
exp_name: name of experiment: {oil, sev_weather, us_accident}
n_trials: total number of trials for Optuna
One run example:

python train.py --exp_name [exp_name] --cuda [cuda:{# of the cuda device available}]
one example:
python train.py --exp_name oil --cuda cuda:0
```

The final evaluation results are saved in a josn file class Final_socres_oil.json, 
when running experiment for the oil dataset.

