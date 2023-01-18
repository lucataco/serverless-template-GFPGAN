import os
import PIL
import cv2
import time
import numpy as np
import base64
import zipfile
from PIL import Image
from io import BytesIO
from gfpgan import GFPGANer
from shutil import make_archive
from realesrgan.utils import RealESRGANer
from models import upsamplers, face_enhancers
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Init is ran on server startup
def init():
    global model
    global upsampler

    #Scaler multiplier, x4 or x2
    multiplier = 4
    model_key = "realesr-general-x4v3"
    print("Init " + model_key)
    upsampler = upsamplers[model_key]
    modelModel = SRVGGNetCompact(**upsampler["initArgs"])
    opt_path = upsampler["path"]
    t = time.time()
    print("Loading " + upsampler["name"])
    #Upsampler model
    upsampler = RealESRGANer(
        scale=multiplier,
        model_path=opt_path,
        model=modelModel,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )
    print("Load time: {:.2f} s".format(time.time() - t))
    model_path = face_enhancers["GFPGAN"]["path"]
    opt_path = model_path
    print("Init GFPGan")
    t = time.time()
    model = GFPGANer(
        model_path=opt_path,
        upscale=multiplier,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )
    print("Load time: {:.2f} s".format(time.time() - t))
    print()


def decodeBase64Image(imageStr: str) -> PIL.Image:
    return PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))


def truncateInputs(inputs: dict):
    clone = inputs.copy()
    if "modelInputs" in clone:
        modelInputs = clone["modelInputs"] = clone["modelInputs"].copy()
        for item in ["input_image"]:
            if item in modelInputs:
                modelInputs[item] = modelInputs[item][0:6] + "..."
    return clone


# Inference is ran for every server call
def inference(all_inputs: dict) -> dict:
    # global model GFPGAN
    global model
    global upsampler

    model_id = "realesr-general-x4v3"
    upsampler = upsamplers[model_id]

    # Parse arguments
    img_byte_str = all_inputs.get('img_bytes', None)

    # Convert to cv2
    nparr = np.fromstring(base64.b64decode(img_byte_str), np.uint8)
    input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    _, _, output = model.enhance(
        input, has_aligned=False, only_center_face=False, paste_back=True
    )
    img_64 = base64.b64encode(cv2.imencode('.jpg', output)[1]).decode()

    return { 'image_base64': img_64 }


if __name__ == "__main__":
    init()

