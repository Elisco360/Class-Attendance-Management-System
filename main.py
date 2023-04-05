from cams_v2 import Attendance as atd

embeddings = "embeddings/faces.npz"
model = "models/students_model.pkl"
database = "database/rcds.csv"

ashesi = atd(database=database,
             model=model,
             face_embeddings=embeddings)

ashesi.home()
