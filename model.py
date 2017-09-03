import csv
import os.path

images = []
angles = []

# Use two folders with images recorded from both tracks
for track in ['Track1', 'Track2']:

    # We have 3 different driving styles -- center of the road, left and right side
    for zone, zone_correction in zip(['Center', 'Left', 'Right'], [.0, .4, -.4]):

        with open('./%s/%s/driving_log.csv' % (track, zone)) as f:
            reader = csv.reader(f)

            # Iterate through all images
            for row in reader:
                side_images = [os.path.basename(row[0]), os.path.basename(row[1]), os.path.basename(row[2])]
                steering_value = float(row[3])

                # Center, left and right images have different steering adjustment values
                for image, side_correction in zip(side_images, [.0, .1, -.1]):

                    # And we create two records, one for original and one for flipped image
                    for sign, prefix in zip([1, -1], ['+', '-']):
                        images.append('%s./%s/%s/IMG/%s' % (prefix, track, zone, image))
                        angles.append(sign * (steering_value + zone_correction + side_correction))

print('Total number of source images is: %d' % len(images))

from keras.models import Sequential
from keras.layers.convolutional import Cropping2D, Conv2D, MaxPooling2D
from keras.layers import Dense, Lambda, Flatten, Dropout, Reshape

# Convert image to grayscale and scale values to fit -1 to 1 range
def image_normalization(x):
    import keras.backend as K
    x *= [.299, .587, .114]
    x = K.sum(x, axis=3)
    x -= 128.
    x /= 255.
    return x

# After image is converted to grayscale it will have one dimention less
def image_output_shape(input_shape):
    return input_shape[:-1]

model = Sequential()

# First we crop the image by removing 65 pixels from the top and 30 from the bottom
model.add(Cropping2D(cropping=((65,30), (0,0)), input_shape=(160,320,3)))
cropping_output_shape = model.layers[-1].output_shape

# Then we call normalization layer
model.add(Lambda(image_normalization, output_shape=image_output_shape))

# We need to shape the result back to 3 dimentions per sample to be able to use Conv2D layers
new_shape = cropping_output_shape[1:-1] + (1,)
model.add(Reshape(new_shape))

# 3 consecutive convolution/maxpooling sequences
model.add(Conv2D(8, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D())

# Flatten layer before doing fully-connected layers
model.add(Flatten())

# 3 consecutive dense/dropout sequences
model.add(Dense(512, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))

# Final layer has linear activation
model.add(Dense(1))

# We're using plain Adam optimizer with mean square error loss function
model.compile(optimizer='adam', loss='mse')

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
images_train, images_test, angles_train, angles_test = train_test_split(images, angles, test_size=0.2)

import cv2
import numpy as np
import sklearn

BATCH_SIZE = 128

# Generator function which returns train/validation batches
def generator(images, angles, batch_size=BATCH_SIZE):
    num_samples = len(images)
    while 1:
        images, angles = shuffle(images, angles)
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]

            image_data = []
            for single_image in batch_images:
                name = single_image[1:]
                img = cv2.imread(name)
                if single_image[:1] == '-':
                    img = cv2.flip(img, 0)
                image_data.append(img)

            X_train = np.array(image_data)
            y_train = np.array(batch_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# We're preparing 3 generators, two when we try different models (with two train/validation sets)
# And the third one for final model training using all the data
train_generator = generator(images_train, angles_train, batch_size=BATCH_SIZE)
validation_generator = generator(images_test, angles_test, batch_size=BATCH_SIZE)
full_generator = generator(images, angles, batch_size=BATCH_SIZE)

```
model.fit_generator(train_generator, steps_per_epoch=\
            len(images_train) / BATCH_SIZE, validation_data=validation_generator,\
            validation_steps=len(images_test) / BATCH_SIZE, epochs=10)
```
# Train model using all the data
model.fit_generator(full_generator, steps_per_epoch = len(images) / BATCH_SIZE, epochs=10)
# Save trained model
model.save('model.h5')
print('Model trained and saved')
