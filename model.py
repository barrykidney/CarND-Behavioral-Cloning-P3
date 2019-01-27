import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda
from keras.layers.convolutional import Conv2D
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_image(img):
    img_crop = img[50:140, :, :]
    img_resized = cv2.resize(img_crop, (200, 66), interpolation=cv2.INTER_AREA)
    img_blur = cv2.GaussianBlur(img_resized, (3, 3), 0)
    return img_blur


def load_driving_log_files(folder_list, path):
    global separator
    lines = []
    for folder in folder_list:
        if Path(path + folder).is_dir():
            with open(path + folder + separator + "driving_log.csv") as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    lines.append(line)
        else:
            print("Directory: \"" + path + "/" + folder + "\" does not exist.")
    return lines


def load_data(lines, path):
    global separator
    images = []
    measurements = []
    for line in tqdm(range(len(lines))):
        for camera in range(3):
            source_path = lines[line][camera]
            split_path = source_path.split("\\")  # Split the path on the Windows separator character
            file_name = split_path[-3] + separator + split_path[-2] + separator + split_path[-1]
            current_path = path + file_name
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = process_image(image)
            if camera == 1:
                measurement = float(lines[line][3]) + 0.2  # left image (1) = steer more to the right (+0.2)
            elif camera == 2:
                measurement = float(lines[line][3]) - 0.2  # right image (2) = steer more to the left (-0.2)
            else:
                measurement = float(lines[line][3])
            images.append(image)
            measurements.append(measurement)
    return images, measurements


def shift_image_horz(img, amt):
    matrix = np.float32([[1, 0, amt], [0, 1, 0]])
    (rows, cols) = img.shape[:2]
    res = cv2.warpAffine(img, matrix, (cols, rows))
    return res


def transform_image(img, mes):
    random_shift = random.randint(-25, 25)
    image_shift = shift_image_horz(img, random_shift)
    measurement_shift = mes + (random_shift * 0.002)
    if random.uniform(0, 1) > 0.5:
        image_shift = cv2.flip(image_shift, 1)
        measurement_shift = measurement_shift * -1
    return image_shift, measurement_shift


def data_generation(imgs, labs, batch, validataion):
    """Generates data containing batch_size samples
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly"""

    # Initialization
    batch_images = np.empty((batch, imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]))
    batch_labels = np.empty((batch, 1))
    # Generate data
    while True:  # loop forever
        for x in range(batch):
            rand = random.randint(0, len(labs)-1)
            if validataion:
                # Store un-altered image and measurement
                batch_images[x] = imgs[rand]
                batch_labels[x] = labs[rand]
            else:
                # Store new image and adjusted measurement
                batch_images[x], batch_labels[x] = transform_image(imgs[rand], labs[rand])
        yield batch_images, batch_labels


def build_test_model():
    m = Sequential()
    m.add(Flatten(input_shape=(160, 320, 3)))
    m.add(Dense(1))
    return m


def build_model():
    m = Sequential()
    m.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(66, 200, 3)))  # normalize & mean center the data
    m.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))  # 24@31X98
    m.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))  # 36@14X47
    m.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))  # 48@5X22
    m.add(Conv2D(64, (3, 3), activation='elu'))  # 64@3X20
    m.add(Conv2D(64, (3, 3), activation='elu'))  # 64@1X18
    #     m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dropout(0.5))
    m.add(Dense(100, activation='elu'))  # 1164
    m.add(Dense(50, activation='elu'))  # 100
    m.add(Dense(10, activation='elu'))  # 50
    m.add(Dense(1))  # 10
    return m


# Parameters
model_name = "model_track12_5mul_5eph"
# separator = '/'
# path_to_data = "/opt/training_data/"
separator = '\\'
path_to_data = "C:\\Users\\barry\\Desktop\\training_data\\"
batch_size = 64
train_valid_split = 0.2
multiplier = 5
number_of_epochs = 5

driving_logs = load_driving_log_files(["track1-1", "track1-2", "recv1-1", "track2-1",
                                       "track2-2", "recv2-1", "recv2-2", "recv2-3"],
                                      path_to_data)

# image_list, measurement_list = load_data(driving_logs, path_to_data)
# X_train = np.array(image_list)
# y_train = np.array(measurement_list)

# model = build_model()
# model.compile(loss='mse', optimizer='Adam')

# model.fit(X_train,
#           y_train,
#           epochs=10,
#           batch_size=32,
#           validation_split=0.2,
#           shuffle=True)

train_logs, valid_logs = train_test_split(driving_logs, test_size=train_valid_split)
print("Number of files loaded:", len(driving_logs))
print("Training set: " + str(len(train_logs)) + ", Validation set: " + str(len(valid_logs)))

print("\nLoading training data")
time.sleep(0.1)
train_image_list, train_measurement_list = load_data(train_logs, path_to_data)
print("Loading validation data")
time.sleep(0.1)
valid_image_list, valid_measurement_list = load_data(valid_logs, path_to_data)

train_gen = data_generation(train_image_list, train_measurement_list, batch_size, False)
valid_gen = data_generation(valid_image_list, valid_measurement_list, batch_size, True)

model = build_model()
model.compile(loss='mse', optimizer='Adam')
history_object = model.fit_generator(train_gen,
                                     validation_data=valid_gen,
                                     validation_steps=514,
                                     epochs=number_of_epochs,
                                     steps_per_epoch=len(train_logs * multiplier) / batch_size)

model.save(model_name + '.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig(model_name + '.png')
plt.close()
