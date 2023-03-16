from cams import Attendance as atd

embeddings = "FR-CAMS\\embeddings\\faces_embeddings.npz"
model = "FR-CAMS\\models\\svm_model_v1.pkl"
database = "FR-CAMS\\database\\records.csv"

ashesi = atd(database=database,
             model=model,
             face_embeddings=embeddings)

ashesi.home()
