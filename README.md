# API for [fontend App](https://github.com/TruongNoDame/vietnamese_htr_frontend/tree/main)

Our repo will provide a Vietnamese handwriting recognition API using FastAPI. The models we use are state-of-the-art models of scene text recognition: [CLIP4STR](https://arxiv.org/abs/2305.14014), [PARSeq](https://arxiv.org/abs/2207.06966) and [ABINet](https://arxiv.org/abs/2103.06495.pdf).

To use our repository, it is recommended to have a GPU for accelerated processing speed (using CPU will be slower)

## 1. Run api

Before running the code, you need to train the model from [this repository](https://github.com/VamosC/CLIP4STR) and place the model file in the corresponding path: `./weight/<model_name>/<file_ckpt>`. Here, the model_name is the name of the model you have selected, and file_ckpt is the name of the model file obtained during training.

Alternatively, you can download and use the pre-trained model files that we have provided. We offer three models: ABINet [download](https://drive.google.com/file/d/1eKAz6DLQNJiUSGNr3uj0UMPHJQmTWJTvE/view?usp=sharing), PARSeq [download](https://drive.google.com/file/d/1eKAz6DLQNJiUSGNruj0UMPHJQmTWJTvE/view?usp=sharing), and CLIP4STR base-32 version [download](https://drive.google.com/file/d/1w-PJVEoXoJ1xBrOhteWawNrDXXZBCfSF/view?usp=sharing).

### Config
Modify the environment variables in the `.env` file to match your selected model as well as the configuration of the API you intend to run.

### Start API
Use the command: 
```
python api.py
```
  
## 2. API Input/Output
The API provides one endpoint: `POST /predict`

Content Type: `application/json`

Request Body:
`image: The image for recognition in base64-encoded string format`

Output:
`text: prediction` Model prediction if successful.
`error detail: error` Error details in case of an error.
