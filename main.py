from cams import Attendance as atd

embeddings = "embeddings/face_embeddings.npz"
model = "models/model_v2.pkl"
database = "database/records.csv"

ashesi = atd(database=database,
             model=model,
             face_embeddings=embeddings)

ashesi.home()
