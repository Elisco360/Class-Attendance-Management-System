import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from mtcnn import MTCNN
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from PIL import Image
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_icon='🎒', page_title='FATTENDANCE', layout="wide")

class DataAugmentation:
    def face_augmentation(path):
        datagen = ImageDataGenerator(
            rotation_range=15,
            shear_range=0.2,
            zoom_range=0.35,
            horizontal_flip=True,
            brightness_range=(0.5, 1.25))

        image_directory = path
        SIZE = 224
        dataset = []
        my_images = os.listdir(image_directory)
        for i, image_name in enumerate(my_images):
            print(image_name)
            if image_name.split('.')[-1] == 'jpg' or image_name.split('.')[-1] == 'jpeg':
                image = io.imread(image_directory + image_name)
                try:
                    image = Image.fromarray(image, 'RGB')
                    image = image.resize((SIZE, SIZE))
                    image = np.array(image)
                    # image = image.reshape((SIZE, SIZE, 1))
                    dataset.append(np.array(image))
                except:
                    image = Image.fromarray(image, 'L')
                    image = image.resize((SIZE, SIZE))
                    image = np.array(image)
                    image = image.reshape((SIZE, SIZE, 1))
                    dataset.append(np.array(image))

        x = np.array(dataset)
        i = 0
        for batch in datagen.flow(x, batch_size=16,
                                  save_to_dir=path,
                                  save_prefix='AU',
                                  save_format='jpg'):
            i += 1
            if i > 4:
                break


class FaceDetection:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y + h, x:x + w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        faces = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                faces.append(single_face)
            except:
                # raise Exception("Face couldn't be detected, retry")
                pass
        return faces

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory + '/' + sub_dir + '/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


embedder = FaceNet()

def get_embedding(face_image):
    face_img = face_image.astype('float32')  # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (None x 160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D image (1x1x512)


def train(old_path):
    faces = FaceDetection(old_path)
    X, Y = faces.load_classes()

    EMBEDDED_X = []
    for img in X:
        EMBEDDED_X.append(get_embedding(img))

    EMBEDDED_X = np.asarray(EMBEDDED_X)

    np.savez_compressed('FR-CAMS\\embeddings\\faces.npz', EMBEDDED_X, Y)

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, Y_train)

    # save the model
    with open('FR-CAMS\\models\\students_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
