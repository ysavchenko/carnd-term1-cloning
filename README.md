# carnd-term1-cloning
Behavioral cloning project for Udacity Self-Driving Car Nanodegree.

This repository includes the code to train and evaluate model for driving both tracks from [Udacity self driving car simulator](https://github.com/udacity/self-driving-car-sim).

This repository does not include training data sets. It expects training data to be collected using Udacity self driving car simulator in training mode with the following folder structure:

- Folders `Track1` and `Track2` for track 1 and track 2 respectively
- Within these folders, three other folders: `Center`, `Left` and `Right` (for center, left edge and right edge driving sessions)

Overall there should be 6 folders:

- `Track1/Center`
- `Track1/Left`
- `Track1/Right`
- `Track2/Center`
- `Track2/Left`
- `Track2/Right`

Here are the links to the main files:

* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* [writeup.md](writeup.md) report summarizing the results
* [track1_20.mp4](track1_20.mp4) and [track2_12.mp4](track2_12.mp4) videos recorded of the model driving on track 1 (speed 20mph) and track 2 (speed 12mph)

Additional files (not required by the rubric points):

* [model\_track_2.h5](model_track_2.h5) model trained on track 2 data only 
* [track1\_from_2.mp4](track1_from_2.mp4) video recorded of the model trained on track 2 steering on track 1 (speed 6mph), see the end of this report for the details