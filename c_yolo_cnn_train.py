'''
Author: BNDou
Date: 2024-04-25 06:23:40
LastEditTime: 2024-04-30 06:24:01
FilePath: \Captchas_BOC\c_yolo_cnn_train.py
Description: 
'''
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def get_random_data(annotation_line, input_shape, max_boxes=4):
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape # (416,416)
    boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    image_resize = image.resize((w, h), Image.BICUBIC)
    box_data = np.zeros((max_boxes,5))
    np.random.shuffle(boxes)
    x_scale, y_scale = float(w / iw), float(h / ih) # 计算验证码图片在横纵方向上缩放的倍数
    for index, box in enumerate(boxes):
        box[0] = int(box[0] * x_scale)
        box[1] = int(box[1] * y_scale)
        box[2] = int(box[2] * x_scale)
        box[3] = int(box[3] * y_scale)
        box_data[index, :] = box
    image_data = np.expand_dims(image_resize, axis=-1)
    image_data = np.array(image_data)/255.
    return image_data, box_data