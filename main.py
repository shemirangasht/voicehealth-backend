from fastapi import FastAPI, File, UploadFile
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):   # 👈 این خیلی مهمه!
    # ذخیره فایل موقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        y, sr = librosa.load(temp_path)
        pitch = librosa.yin(y, fmin=80, fmax=300, sr=sr)
        avg_pitch = float(np.mean(pitch))

        jitter = float(np.std(np.diff(pitch)) / avg_pitch)
        shimmer = float(np.std(y))

        os.remove(temp_path)

        return {
            "pitch": round(avg_pitch, 2),
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4)
        }
    except Exception as e:
        return {"error": str(e)}
