#!/usr/bin/env bash
apt-get update
apt-get -y install python-pip
apt-get -y install ipython ipython-notebook
pip install numpy
pip install opencv-python
pip install -U scikit-learn
pip install --upgrade tensorflow-gpu
pip install keras
pip install pandas
pip install scipy
pip install tsne