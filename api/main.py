from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

import uvicorn
app = FastAPI()

@app.get("/ping")
async def ping():
    return "server is ok"

@app.post("/predict")
async def predict(file: UploadFile):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)