from typing import Union
import shutil
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import io

from fastapi import FastAPI, UploadFile, File


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/audio")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    y, sr = librosa.load(io.BytesIO(contents), sr = None)
    segment_length = 5

    num_segments = int(np.ceil(len(y) / float(segment_length * sr)))
    features = []

    # Extract features for each segment
    for i in range(num_segments):
        # Calculate start and end frame for the current segment
        start_frame = i * segment_length * sr
        end_frame = min(len(y), (i + 1) * segment_length * sr)

        # Extract audio for this segment
        y_segment = y[start_frame:end_frame]

        # Extract features
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y_segment))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
        mfccs = librosa.feature.mfcc(y=y_segment, sr=sr)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Append the extracted features to the list
        features.append([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, *mfccs_mean])

    df = pd.DataFrame(features)

    n_fft=2048
    hop_length=512
    n_mels=128
    mels = librosa.feature.melspectrogram(y=y, sr=sr,  n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(mels, ref=np.max)
    plt.axis("off")
    plt.tight_layout()
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the figure to free memory
    buf.seek(0)
    image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(image, (200, 200))
    rescaled_img = resized_img / 255.0
    img_to_predict = np.expand_dims(rescaled_img, axis=0)
    

    featuresModel = tf.keras.models.load_model('featuresModel.h5')
    imageModel = tf.keras.models.load_model('imageModel.h5')
    pred = imageModel.predict(img_to_predict)
    print(pred)
    imagePred = 1 - pred[0][0]
    resultArr = featuresModel.predict(df)
    featurePred = sum(resultArr) / len(resultArr)
    featurePred = featurePred[0]
    final_result = np.mean([featurePred, imagePred], axis=0)
    if(final_result > 0.7):
        return{"Voice" : f"AI with probablilty: {final_result}"}
    else:
        return{"Voice" : f"Real with probablilty: {1 - final_result}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)