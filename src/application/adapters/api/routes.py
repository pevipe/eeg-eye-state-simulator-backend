# Create fastAPI router

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.application.adapters.persistence.repository import get_loaded_subjects, is_optimized_subject_for_algorithm
from src.application.core.feature_extraction.data_loaders import upload_subject

app = FastAPI()


@app.get("/uploaded_subjects")
def list_of_uploaded_subjects():
    names_list = get_loaded_subjects()
    return JSONResponse(content={"data": names_list})


@app.post("/upload_subject")
async def upload_dataset(file: UploadFile = File(...)):
    name = file.filename.rsplit('.csv')[0]
    content = await file.read()
    upload_subject(name, content)
    return JSONResponse(content={"message": f"Subject {name} loaded successfully"}, status_code=200)


@app.post("/optimize")
async def optimize_subject_algorithm(subject: str, algorithm: str, window: int):
    pass


@app.get("/is-optimized")
def is_optimized_subject(subject: str, algorithm: str, window: int):
    res = is_optimized_subject_for_algorithm(subject, algorithm, window)
    return JSONResponse(content={"is_optimized": res})


@app.post("/train")
def train_algorithm(subject: str, algorithm: str, window: int, train_set_size: int, use_custom_hyperparams: bool):
    pass
