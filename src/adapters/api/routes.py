import json

from fastapi import APIRouter, File, UploadFile, Response
from fastapi.responses import JSONResponse

from src.adapters.persistence.repository import get_loaded_subjects, is_optimized_subject_for_algorithm
from src.adapters.persistence.repository import upload_subject
from src.application.usecases import optimize_for_algorithm, make_windows, train_algorithm

router = APIRouter(tags=["Classifiers"])


@router.get("/uploaded_subjects")
def list_of_uploaded_subjects():
    names_list = get_loaded_subjects()
    return JSONResponse(content={"data": names_list})


@router.post("/upload_subject")
async def upload_dataset(file: UploadFile = File(...)):
    name = file.filename.rsplit('.csv')[0]
    print("Uploading subject:", name)
    content = await file.read()
    return upload_subject(name, content)


@router.post("/window_subject")
async def window_subject(subject: str, window: int):
    print("Windowing subject:", subject, "with window size:", window)
    make_windows(subject, window)
    return JSONResponse(content={"message": "Subject windowed successfully"}, status_code=200)


@router.get("/is_optimized")
def is_optimized_subject(subject: str, algorithm: str, window: int):
    res = is_optimized_subject_for_algorithm(subject, algorithm, window)
    return res


@router.post("/optimize")
async def optimize_subject_algorithm(subject: str, algorithm: str, window: int):
    print("Optimizing subject:", subject, "with algorithm:", algorithm, "and window size:", window)
    if algorithm not in ['AdaBoost', 'DecisionTree', 'kNN', 'LDA', 'RandomForest', 'QDA', 'SVM']:
        return JSONResponse(content={"message": "Algorithm not recognized."}, status_code=400)
    else:
        m = optimize_for_algorithm(subject, algorithm, window)
        return JSONResponse(content={"message": m}, status_code=200)


@router.post("/train")
async def train(subject: str, algorithm: str, window: int, train_set_size: int, use_optimized_hyperparams: bool):
    results = train_algorithm(subject, algorithm, window, train_set_size, use_optimized_hyperparams)
    results['real_tags_in_test'] = results['real_tags_in_test'].tolist()
    results['predicted_tags_in_test'] = results['predicted_tags_in_test'].tolist()

    return Response(content=json.dumps(results), media_type="application/json")
