# Jupyter notebooks to explain Reservoir Computing
Reservoir Computing (RC) is a paradigm for training recurrent neural networks. Echo State Networks (ESNs) is the most famous RC framework: [more info on Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network). This repository is based on ESNs.

## Install guide

#### Jupyter notebook
- See detailed info here: http://jupyter.readthedocs.io/en/latest/install.html
- Or, if you do no have Python with scientific librairies installed yet, install Anaconda for Python 3: https://www.anaconda.com/download
- If you already have Python 3 with scientific librairies (Numpy, Scipy, Matplotlib, ...): Open a terminal (Linux / Mac OS X) and enter the following commands: (no need of "sudo" under Mac OS X)
```sudo pip3 install --upgrade pip
sudo pip3 install jupyter
```

#### Widgets for Jupyter Notebook
- Enter the following commands in a terminal:
```sudo pip3 install ipywidgets
sudo jupyter nbextension enable --py widgetsnbextension
```
- Details here ipywidgets.readthedocs.io/en/stable/user_install.html

## Run the notebooks on your computer
- clone the repository:
```git clone https://github.com/neuronalX/Reservoir-Jupyter.git
```
- run Jupyter (this opens a Notebook in a browser tab):
```jupyter notebook
```
- in your browser, select one of the \*.ipynb files to run the corresponding notebook.
- to begin, start with **Minimal_ESN_-_FR.ipynb** (in French) or **Minimal_ESN_-\_EN.ipynb** (in English).

## Run the notebooks online
Press this button to load (previous versions of) the notebooks online:
[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/romainpastureau/reservoir-jupyter)

## About
This repository was first created by @RomainPastureau during his internship with @neuronalX and @rougier at Inria (Research Institute for Mathematics and Computer Science) in Bordeaux, France in June 2016.
It is now pursued by @neuronalX.
