from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
import tensorflow as tf
import numpy as np

app = FastAPI()

path_to_model = 'E:\Hamza\AI\Potato Disease Classification\Models\potato classification model.h5'
model = tf.keras.models.load_model(path_to_model)

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Potato Disease Identification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmirmltdeIgnarGtHrAhEBaUuFujTCFSuwsA&usqp=CAU");
            background-repeat: no-repeat;
            background-position: center;
            background-size: cover;
            color: blue; /* Set text color to white */
            text-align: center; /* Center-align text */
            font-size: 25px;
        }
        .container {
            padding: 2em; /* Add padding to center content */
        }
        .appbar {
            background: #be6a77;
            box-shadow: none;
            color: white;
        }
        .mainContainer {
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmirmltdeIgnarGtHrAhEBaUuFujTCFSuwsA&usqp=CAU");
            background-repeat: no-repeat;
            background-position: center;
            background-size: cover;
            height: 93vh;
            margin-top: 8px;
        }
        .gridContainer {
            justify-content: center;
            padding: 4em 1em 0 1em;
        }
        .imageCard {
            margin: auto;
            max-width: 400px;
            height: 500px;
            background-color: transparent;
            box-shadow: 0px 9px 70px 0px rgb(0 0 0 / 30%) !important;
            border-radius: 15px;
        }
        .imageCardEmpty {
            height: auto;
        }
        .noImage {
            margin: auto;
            width: 400px;
            height: 400px !important;
        }
        .content {
            padding: 1rem;
        }
        .uploadIcon {
            background: white;
        }
        .tableContainer {
            background-color: transparent !important;
            box-shadow: none !important;
        }
        .table {
            background-color: transparent !important;
        }
        .tableHead {
            background-color: transparent !important;
        }
        .tableRow {
            background-color: transparent !important;
        }
        .tableCell {
            font-size: 22px;
            background-color: transparent !important;
            border-color: transparent !important;
            color: #000000a6 !important;
            font-weight: bolder;
            padding: 1px 24px 1px 16px;
        }
        .tableCell1 {
            font-size: 14px;
            background-color: transparent !important;
            border-color: transparent !important;
            color: #000000a6 !important;
            font-weight: bolder;
            padding: 1px 24px 1px 16px;
        }
        .tableBody {
            background-color: transparent !important;
        }
        .text {
            color: white !important;
            text-align: center;
        }
        .buttonGrid {
            max-width: 416px;
            width: 100%;
        }
        .detail {
            background-color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .loader {
            color: #be6a77 !important;
        }
        .clear-button {
            background-color: #000000a6;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <header>
        <h1>Potato Disease Identification</h1>
    </header>
       <div class="container">
        <form class="upload-form" id="image-upload" action="/predict" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file" id="file" class="input">
            <input type="submit" value="Upload Image" name="submit" class="upload-button">
            <br>
            <button class="clear-button" type="button" id="clear-button" style="display:none;">Clear Image</button>
        </form>
        <img id="image-preview" src="#" alt="Uploaded Image" class="imageCard">
        <div id="prediction-result" class="detail"></div>
    </div>
    <script>
        document.getElementById("file").addEventListener("change", function (event) {
            const imagePreview = document.getElementById("image-preview");
            const clearButton = document.getElementById("clear-button");
            const file = event.target.files[0];
            const reader = new FileReader();

            reader.onload = function () {
                imagePreview.src = reader.result;
                clearButton.style.display = "block"; // Show the clear button
            };

            if (file) {
                reader.readAsDataURL(file);
            } else {
                imagePreview.src = "#";
                clearButton.style.display = "none"; // Hide the clear button
            }
        });

        document.getElementById("clear-button").addEventListener("click", function () {
            const imagePreview = document.getElementById("image-preview");
            const clearButton = document.getElementById("clear-button");
            const fileInput = document.getElementById("file");

            imagePreview.src = "#";
            clearButton.style.display = "none";
            fileInput.value = ""; // Clear the file input
        });
    </script>
</body>
</html>

"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=html_template)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))

        image = image.resize((256, 256))

        image = np.array(image)  # Normalize to [0, 1]

        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        prediction_confidence = np.max(prediction)
        prediction_name = np.argmax(prediction)
        prediction_name = class_names[prediction_name]

        response_message = f"Your potato plant has/is: '{prediction_name}'<br>"
        response_message += f"The confidence of this prediction is: {prediction_confidence:.4f}"

        return HTMLResponse(content=response_message)
    except Exception as e:
        return {'error': str(e)}