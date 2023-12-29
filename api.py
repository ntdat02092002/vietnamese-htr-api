from fastapi import FastAPI, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import json
import numpy as np
import torch
from PIL import Image
from torch import inference_mode
import base64
from io import BytesIO

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

DEVICE = "cuda"
CHECKPOINT = "./weight/vl4str/epoch=87-step=179942-val_accuracy=90.1811-val_NED=96.4276.ckpt"

def load_model():
    print("load model")
    model = load_from_checkpoint(CHECKPOINT, **{}).eval().to(DEVICE)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    print("load model successfully")
    return model, img_transform

model, img_transform = load_model()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
     CORSMiddleware,
     allow_origins=origins,
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

def process_image(base64_data):
    try:
        # Loại bỏ tiền tố 'data:image/png;base64,'
        base64_data = base64_data.split(',')[1]

        # Giải mã base64 thành dữ liệu nhị phân
        binary_data = base64.b64decode(base64_data)

        # Đọc ảnh từ dữ liệu nhị phân
        image = Image.open(BytesIO(binary_data))

        # Chuyển định dạng ảnh thành 'RGB' nếu nó không phải là 'RGB'
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/")
async def home(request: Request):
    return {message: "hello"}

@app.post("/predict/")
async def process_image_endpoint(request: Request):
    try:
        # Lấy chuỗi base64 từ dữ liệu JSON
        data_url = await request.json()
        base64_data = data_url.get("image", "")

        # Xử lý ảnh và nhận dữ liệu ảnh xử lý
        processed_image_data = process_image(base64_data)

        image = img_transform(processed_image_data).unsqueeze(0).to(DEVICE)
        with inference_mode():
            p = model(image).softmax(-1)
            pred, p = model.tokenizer.decode(p)

        return JSONResponse(content={"text": pred[0].strip()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    app_str = 'api:app'
    uvicorn.run(app_str, host='0.0.0.0', port=8080, reload=True, workers=1)