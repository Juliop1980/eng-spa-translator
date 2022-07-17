#Natural Language Processing 

English-Spanish Machine Translation

## About the project
Machine Translation (MT) or automated translation is a process when a computer software translates text from one language to another without human involvement. Today, machine translation works best in scenarios where a text needs to be conveyed in an understandable form in another language. Using deep neural networks, the performance and accuracy of machine translation models have been increased significantly in recent years.

This project is a machine translation model to translate English text into Spanish and vice versa.


### Features

Currently, the Scripts contained in this folder allow to:


Note:
The instructions are written for Linux, specifically Ubuntu 20.04.4 LTS 

### Development
The scripts are currently under development. The development is done using *Python 3.8.10*. The rest of the requirements can be found in the [requirements file (requirements.txt)](requirements.txt).

## Set up

### Starting 🚀

_
These instructions will allow you to get a copy of the project running on your local machine for development and testing purposes_

Look at **Deployment** to know how to deploy this project.


### Requirements 📋

_**Git**_

```
sudo apt update
sudo apt install git
```

_**Python 3.8.10**_
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8
```

_**Install Pip**_
```
sudo apt install python3-pip
```


**virtualenv**

_install python3-venv with_
```
sudo apt install python3.8-venv
```

_create a directory to save virtual environments and enter it_
```
mkdir virtualenvs
cd virtualenvs
```

_Set the environment_
```
python -m venv env1
```

_To activate use_
```
source env1/bin/activate
```

_To deactivate it use_
```
source env1/bin/deactivate
```

### Installation 🔧

_Follow these steps once done with the **Requirements**:_

_**NOTE: Keep your virtual environment activated for the installation.**



_clone the github repository_

```
git clone https://git.tu-berlin.de/juliop1996/nlp_english-spanish_machine_translation.git
```

_enter repository_

```
cd nlp_english-spanish_machine_translation/
```




_A requirements file was generated that will allow the automatic installation of the modules with_

```
pip install -r requirements.txt
```

## Run and coding ⌨️

### Structure of the code


#### insights_of_data






### Important commands

_to run the scripts_

They take around 5 minutes each.

```
python insights_data.py
```


## Authors

Group 22

-Anne Laure Olga Tettoni\
-Julio Cesar Perez Duran