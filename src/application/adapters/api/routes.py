from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.application.adapters.persistence.repository import get_loaded_subjects, is_optimized_subject_for_algorithm
from src.application.adapters.persistence.repository import upload_subject
from src.application.core.run_algorithms import optimize_for_algorithm, make_windows

router = APIRouter(tags=["Classifiers"])


@router.get("/uploaded_subjects")
def list_of_uploaded_subjects():
    names_list = get_loaded_subjects()
    return JSONResponse(content={"data": names_list})


# @router.post("/upload_subject")
# async def upload_dataset(file: UploadFile = File(...)):
#     name = file.filename.rsplit('.csv')[0]
#     content = await file.read()
#     m = upload_subject(name, content)
#     return m
#     # return JSONResponse(content={"message": m}, status_code=200)


# from fastapi.middleware.cors import CORSMiddleware
# CORS middleware to allow requests from Angular
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://tu-aplicacion-angular.com"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )


#
# @router.post("window_subject")
# async def window_subject(subject: str, window: int):
#     make_windows(subject, window)
#     return JSONResponse(content={"message": "Subject windowed successfully"}, status_code=200)
#
#
# @router.post("/optimize")
# async def optimize_subject_algorithm(subject: str, algorithm: str, window: int):
#     if algorithm not in ['AdaBoost', 'DecisionTree', 'KNN', 'LDA', 'RandomForest', 'QDA', 'SVM']:
#         return JSONResponse(content={"message": "Algorithm not recognized."}, status_code=400)
#     else:
#         m = optimize_for_algorithm(subject, algorithm, window)
#         return JSONResponse(content={"message": m}, status_code=200)
#
#
# @router.get("/is-optimized")
# def is_optimized_subject(subject: str, algorithm: str, window: int):
#     res = is_optimized_subject_for_algorithm(subject, algorithm, window)
#     return JSONResponse(content={"is_optimized": res})
#
#
# @router.post("/train")
# def train_algorithm(subject: str, algorithm: str, window: int, train_set_size: int, use_optimized_hyperparams: bool):
#     return JSONResponse(content={"message": "Not implemented. Coming soon!"}, status_code=501)
