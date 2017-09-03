#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[crop]: ./images/crop.jpg "Image Crop"
[model]: ./images/model.png "Model Visualization"
[track_1_center]: ./images/track_1_center.jpg "Track 1 Center"
[track_1_left]: ./images/track_1_left.jpg "Track 1 Left"
[track_1_right]: ./images/track_1_right.jpg "Track 1 Right"
[track_2_center]: ./images/track_2_center.jpg "Track 2 Center"
[track_2_left]: ./images/track_2_left.jpg "Track 2 Left"
[track_2_right]: ./images/track_2_right.jpg "Track 2 Right"
[track_2_left_flipped]: ./images/track_2_left_flipped.jpg "Track 2 Left Flipped"
[training]: ./images/training.png "Training Loss"

## Rubric Points
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* [model.py](model.py) containing the script to create and train the model
* [drive.py](drive.py) for driving the car in autonomous mode
* [model.h5](model.h5) containing a trained convolution neural network 
* [writeup.md](writeup.md) summarizing the results (this file)
* [track1_20.mp4](track1_20.mp4) and [track2_12.mp4](track2_12.mp4) videos recorded of the model driving on track 1 (speed 20mph) and track 2 (speed 12mph)

Additional files (not required by the rubric points):

* [model\_track_2.h5](model_track_2.h5) model trained on track 2 data only 
* [track1\_from_2.mp4](track1_from_2.mp4) video recorded of the model trained on track 2 steering on track 1 (speed 6mph), see the end of this report for the details

####2. Submission includes functional code
Using the Udacity provided simulator and my [drive.py](drive.py) file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 8 and 32 (model.py lines 62-67). After 3 convolution and 3 max pooling layers I have 3 dense layers and then a final layer with a single output.

All layers (except for the last one) have RELU activation logic to add non-linearity to the model. The data is normalized in the model using a Keras lambda layer (code lines 55 and 37-42). 

####2. Attempts to reduce overfitting in the model

The model contains 3 dropout layers in order to reduce overfitting (model.py lines 74, 76 and 78). Also reduce overfitting 3 max pooling layers after each convolution layer (lines 63, 65, and 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 125-127, commented now). The number of epochs (10) was selected based on the fact that validation loss started increasing after the 10th epoch.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 84).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Project recommendations were to combine center driving with returning from the side back to the center of the road, but I decided to use slightly different approach. I did record several laps of central driving, but instead of recording recovery I've recorded one lap driving on the right edge of the road and one lap -- on the left side.

For details about how ended up with this data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

Model architecture was inspired by LeNet and similar image recognition architectures. Basically it is a combination of several layers of convolution-maxpooling followed by several dense layers. I did try to use NVIDIA architecture as well as VGG-16, but the one I came up with showed good results and was quite qiuck to train, so I kept this architecture with some tweaks (discussed later).

The most important change was converting images to grayscale before running them through the network. I've tried it after several days of testing model on the track and it improved results a lot (before the model was barely driving on the first track, but after converting to grayscale it could almost finish track 2). Also it reduced the size of the model, because less input meant less need for convolution depth, I've reduced convolution depth from 24-to-96 depth to 8-to-32 without noticable increase in training/validation loss and model performance on the track.

Other things I've tuned were size of the crop image (I've tried to make is as small as possible, but still maintaining the model performance). Below you can find the illustration of the final crop parameters I ended up using (65 pixels from the top and 30 from the bottom). And there was a lot of trial and error with adding adjustment to the training data (depending on where I was driving), but I'll discuss it in the next session.

![crop]

Overfitting was not really a problem, a combination of max pooling after each convolution layer and dropout layers after each dense layer, which I've added from the start, prevented the model from overfitting.

####2. Final Model Architecture

The final model architecture (model.py lines 48-81) consisted of a convolution neural network (3 convolution layers with 5x5 kernel size each followed by 2x2 max pooling layer, depth from 8 in the first layer to 32 in the last) followed by 3 dense layers (with width 512, 256 and 64 respectively) each followed by a dropout layer. And the final layer is a single node with linear activation (all other layers have a RELU activation).

Here is a visualization of the architecture:

![model]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two-three laps on both tracks using center lane driving. Here is an example image of center lane driving (for both tracks):

![track_1_center] ![track_2_center]

Then as per project recommendations I've recorded recovery from the sides back to the center of the track. This took a while, because I could not record it continuously (otherwise the model will not only learn to return to the center, but to go off center too). So I stopped recording, went to the side, started recording and pulled back to the center.

Training model on this dataset turned out to be a disaster. Several models I've tried initially (including NVIDIA model recommended in the project materials and VGG-16 network I chose for its simplicity) even stuck in a local minimum returning average steering value for any input data.

At first I blamed the model, tried different architectures, optimizers, activation layers etc. But nothing helped. Then I looked closely to the training data and found the problem with it: I used keyboard for steering, which moved steering all the way to the left or right when left or right button is pushed and suddenly back to zero when direction button is released. In the end the model had to generalize similarly looking images, but steering value was 0.3 on one and 0 on another (simply because it was the moment I released direction button). No wonder it chose to return average value as a result, there was no way to generalize this data efficiently.

So I've thrown away all recorded data and decided to record everything again, but this time using trackpad for smooth steering. And to save time on recording recovery to the center of the road (the one which required constant pausing) I've decided to try and record driving on the right and on the left edge of the road and put such recordings to separate folders.

Here are some examples of right side and left side driving on both tracks.

![track_1_right] ![track_2_right]
![track_1_left] ![track_2_left]

So in the end I had 3 images of each moment for center driving (recorded by center, left and right camera) and 6 images for driving on the side of the road (3 on the left and 3 on the right). What I've decided to do is to train model on all of them with small adjustment to the steering value for the driving on the edge of the track. I ended up adjusting steering by 0.1 for left/right camera on the vehicle and by 0.4 for left/right side. So basically on every point of the track I had 9 images with the following adjustments

Left | Center | Right
---|---|---
`0.5` `0.4` `0.3` | `0.1` `0` `-0.1` | `-0.3` `-0.4` `-0.5`

To increase the size of the dataset I've also added horizontal flipping of the source images with changing the sign of the steering. The sign of the adjustment had to be changed too.

Here is an example of how such flipping looks like:

![track_2_left] ![track_2_left_flipped]

At the end with three recording sessions on both tracks (6 sessions in total) and flipping each image to get the extra data I had 221874 data points to train my model on. I shuffled the data randomly, dedicated 20% of it to the validation set and started training the model using Adam optimizer. 

Here is how the loss when training for 20 epochs looked like:

![training]

Validation loss is lower in every epoch because of the dropout layers (they drop some of the signals during testing, but pass all of them during validation).

As you can see, after epoch 10 validation loss stopped declining, so the final model was trained on 10 epochs with a resulting loss arount 0.01.

## Bonus Section
While not the part of the rubric points, here are some things I wanted to include additionally to my report.

###Final Model Performance

####Model trained on both tracks

Below you can see two videos (YouTube versions with links to original below each video) of the model steering on track 1 and track 2. The model was sensitive to the vehicle speed, on both tracks my computer was not powerful enough to calculate steering on each frame, so model went off the track on high speed. Videos below are recorded at speed 20 on track 1 and speed 12 on track 2. When testing the final model please use speeds not higher than I used.

[![Track 1, Speed 20](http://img.youtube.com/vi/pyk1e-srEeM/0.jpg)](http://www.youtube.com/watch?v=pyk1e-srEeM)

Original: [Track 1, Speed 20](track1_20.mp4)

[![Track 2, Speed 12](http://img.youtube.com/vi/T1bnMidPC_k/0.jpg)](http://www.youtube.com/watch?v=T1bnMidPC_k)

Original: [Track 2, Speed 12](track2_12.mp4)

####Model trained on track 2

As a way of displaying generalization abilities of the model I've also tried model trained on track 2 only to steer on track 1. Here is the video of how it performed:

[![Track 1, Speed 20](http://img.youtube.com/vi/3Gvfao6V6wM/0.jpg)](http://www.youtube.com/watch?v=3Gvfao6V6wM)

Original: [Track 1 (trained on track 2), Speed 6](track1_from_2.mp4)

The model was not able to drive on its own, there are several places where the video jumps, this is where I help it to get back on the track. 

Even considering the fact that model needed help, still it showed quite impressive results. Remember, this is the first time it sees this track and track 2 was much different (it has a clear lane separator on the whole track and road edges looked differently). But still the model was able to drive and needed my help only 4 or 5 times. I think this is a good indicator that I chose a correct architecture and training technique.
