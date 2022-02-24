from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()
Model = tf.keras.models.load_model("../saved_models/1")
Class_names = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "server is ok"

#convert image to numpy array for prediction
def read_file_as_image(data) -> np.ndarray:
    #data coming as bytes for manipulating that BytesIO. PIL pillow helps to open image data then convert as numpy array and return
    img = Image.open(BytesIO(data))
    resize_img = img.resize((256,256))
    print(resize_img.size)

    return np.array(resize_img)
    #return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict(file: UploadFile):
    image = read_file_as_image(await file.read())
    #our model rad the image as batch so
    img_batch = np.expand_dims(image, 0) #crteate [[]]
    prediction = Model.predict(img_batch) #prediction [[.5597,0.0052,546].]
    predicted_class = Class_names[np.argmax(prediction[0])]
    confidance = np.max(prediction[0])

    return {
        'class': predicted_class,
        'confidence': float(confidance)
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)