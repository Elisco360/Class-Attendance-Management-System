from cams import Attendance as atd

embeddings = "embeddings\\faces_embeddings.npz"
model = "models\\svm_model_v1.pkl"
database = "database\\records.csv"

ashesi = atd(database=database,
             model=model,
             face_embeddings=embeddings)

ashesi.home()
