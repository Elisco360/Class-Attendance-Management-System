from train import DataAugmentation, train
import streamlit as st
from PIL.Image import Image
from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from streamlit_option_menu import option_menu
from datetime import date
import time
import os
from pathlib import Path

class Attendance:
    def __init__(self, database, face_embeddings, model):
        self.database = Attendance.__get_database__(database)
        self.embeddings = Attendance.__get_features__(face_embeddings)
        self.model = Attendance.__get_model__(model)

        self.detector = MTCNN()
        self.facenet = FaceNet()
        self.encoder = LabelEncoder()

        self.encoder.fit(self.embeddings)
        self.encodings = self.encoder.transform(self.embeddings)

        self.old_path = 'database/'

    @staticmethod
    # @st.cache_data
    def __get_model__(model):
        m = pickle.load(open(model, 'rb'))
        return m

    @staticmethod
    # @st.cache_data
    def __get_features__(emb):
        embed = np.load(emb)['arr_1']
        return embed

    @staticmethod
    def __get_database__(data):
        db = pd.read_csv(data)
        return db

    def add_student(self, name, id):
        self.database.loc[len(self.database.index)] = [name, str(id), 'Calculus', 'Absent 📕', '', '']
        self.database.to_csv('database/rcds.csv', index=False, header=True)

    def save_image(self, image_array, label, path):
        label_path = Path(path) / label
        label_path.mkdir(parents=True, exist_ok=True)

        # create image object from the numpy array
        image = Image.fromarray(np.uint8(image_array))

        # save the image to the label's directory
        image_path = label_path / "image.jpg"
        image.save(str(image_path))
        return label_path

    def home(self):
        st.markdown("<h1 style='text-align: center'>CLASS ATTENDANCE MANAGEMENT SYSTEM</h1>", unsafe_allow_html=True)
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        with st.sidebar:
            st.header("Menu")
            option = option_menu('', ['📸 Take Attendance', '🧔🏾🧑🏾‍🦰 Register Face'],
                                 icons=["nothing", "nothing"],
                                 menu_icon='nothing',
                                 orientation='vertical')
        if option == "📸 Take Attendance":
            self.__get_face()
        elif option == "🧔🏾🧑🏾‍🦰 Register Face":
            st.header('Register Face')
            st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            st.info('The registration will require you to take a picture of yourself without any facial expression, '
                    'extra accessories that could prevent the camera from capturing every bit of your face and '
                    'finally good lighting. Look into the camera buddy, we need a raw picture of your face 🙈🙊 ')
            with st.expander('Open to register face'):
                st.markdown("---")
                l,r = st.columns(2)
                std_name = l.text_input('Name')
                std_id = r.text_input('Student ID')
                picture = st.camera_input("Don't smile or frown okay 🥲")

                if picture:
                    img: Image = Image.open(picture)
                    img_array = np.array(img)
                    p = self.save_image(img_array, std_name, self.old_path)
                    DataAugmentation.face_augmentation(p)
                    with st.spinner('Hang tight, you are being registered'):
                        train(self.old_path)
                    self.add_student(std_name, std_id)
                    st.balloons()
                    st.success("You've been registered")

    def get_courses(self):
        my_courses = set()
        for i in self.database["Class"]:
            if "/" in i:
                sep = i.split("/")
                for j in sep:
                    my_courses.add(j)
            else:
                my_courses.add(i)
        return my_courses

    def __get_face(self):
        st.header('Take Attendance')
        st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        cam, records = st.tabs(['✅ Take Attendance', '📊 Attendance Records'])

        with cam:
            with st.expander("Open Camera and take picture"):
                st.markdown("---")
                picture = st.camera_input("Enjoy class today and be great 😉✌️")

            if picture:
                img: Image = Image.open(picture)
                img_array = np.array(img)
                detected_face = self.__detect_face__(img_array)
                self.__recognize_face__(detected_face)

        with records:
            st.header("📊 Attendance Records")
            st.markdown("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            self.__display_database()
            # l,r = st.columns(2)
            # l.download_button('Download file', self.database)
            reset = st.button("Reset Database")
            if reset:
                self.__reset_database__()

    def __detect_face__(self, face):
        detected_face = None

        try:
            cv_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            x, y, w, h = self.detector.detect_faces(cv_face)[0]['box']
            detected_face = cv_face[y:y + h, x:x + w]
            detected_face = cv2.resize(detected_face, (160, 160))
            detected_face = np.expand_dims(detected_face, axis=0)
            # st.image(detected_face)
        except:
            pass

        return detected_face

    def __recognize_face__(self, detected_face):
        if detected_face is not None:
            face_embedding = self.facenet.embeddings(detected_face)
            # probability = self.model.predict_proba(face_embedding)
            prediction = self.model.predict(face_embedding)
            student_name = self.encoder.inverse_transform(prediction)[0]
            st.success(f"✅ Attendance for {student_name} has been marked.")
            self.__update_database__(student_name)
        else:
            pass

    def __update_database__(self, name):
        # df.loc[df['name'] == student_name, 'status'] = 'Yes'
        self.database.loc[self.database['Name'] == name, 'Status'] = "Present 📗"
        self.database.loc[self.database['Name'] == name, 'Date'], t = date.today(), time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        self.database.loc[self.database['Name'] == name, 'Time'] = current_time
        self.database.to_csv('database/rcds.csvv', index=False, header=True)

    def __reset_database__(self):
        self.database['Status'] = "Absent 📕"
        self.database['Date'] = ''
        self.database['Time'] = ''
        self.database.to_csv('database/rcds.csv', index=False, header=True)

    def __display_database(self):
        st.dataframe(self.database, use_container_width=True)
