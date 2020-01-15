# MuteNRossi

## Introduction
MuteNRossi is an application for American Sign Language recognition elaborated by students of Computer Vision in UPM.

![Logo](bin/muten.png | width=150)

## Instructions
1. Run main.py
```
python3 main.py
```
2. Stand at a distance of 40 cm. from the camera, watching your head in the left part of the screen.
3. Press 'b' to capture a new background.
4. Raise your palm so it fits into the 2 green regions, and then, press 's' to capture the skin.
5. Enjoy.

## Instalation
You will have to install all libraries being used by the program. To do so, we have add a file named `requirements.txt` which will help you on this task.

1. Install python library `virtualenv`
```
pip install virtualenv
```

2. Install the libraries on your virtualenv, or in your global libraries if prefered.
```
pip install -r requirements.txt
```

3. Run your virtual environment if you have installed the libraries and try running the program.

## Dataset
You can find the dataset used to train and test this model on the following link.

[Used dataset](https://www.kaggle.com/grassknoted/asl-alphabet)
