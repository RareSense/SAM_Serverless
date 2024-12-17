import modal
from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from fastapi import FastAPI
from models import ImageRequest, ImageResponse, Response
from fastapi import HTTPException
import time
from utils import base64_to_image, image_to_base64,run_sam
from config import logger
import numpy as np

# CUDA version and dependencies
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Define the Modal image with dependencies
mask_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
    "annotated-types==0.7.0",
    "anyio==4.7.0",
    "certifi==2024.12.14",
    "charset-normalizer==3.4.0",
    "click==8.1.7",
    "coloredlogs==15.0.1",
    "contourpy==1.3.1",
    "cycler==0.12.1",
    "fastapi==0.115.6",
    "filelock==3.16.1",
    "flatbuffers==24.3.25",
    "fonttools==4.55.3",
    "fsspec==2024.10.0",
    "h11==0.14.0",
    "httpcore==1.0.7",
    "httpx==0.28.1",
    "humanfriendly==10.0",
    "idna==3.10",
    "Jinja2==3.1.4",
    "kiwisolver==1.4.7",
    "MarkupSafe==3.0.2",
    "matplotlib==3.10.0",
    "mpmath==1.3.0",
    "networkx==3.4.2",
    "numpy==2.2.0",
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cudnn-cu12==9.1.0.70",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147",
    "nvidia-cusolver-cu12==11.6.1.9",
    "nvidia-cusparse-cu12==12.3.1.170",
    "nvidia-nccl-cu12==2.21.5",
    "nvidia-nvjitlink-cu12==12.4.127",
    "nvidia-nvtx-cu12==12.4.127",
    "onnx==1.17.0",
    "onnxruntime==1.20.1",
    "opencv-python==4.10.0.84",
    "packaging==24.2",
    "pillow==11.0.0",
    "protobuf==5.29.1",
    "pycocotools==2.0.8",
    "pydantic==2.10.3",
    "pydantic_core==2.27.1",
    "pyparsing==3.2.0",
    "python-dateutil==2.9.0.post0",
    "requests==2.32.3",
    "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf#egg=segment_anything",
    "setuptools==75.6.0",
    "six==1.17.0",
    "sniffio==1.3.1",
    "starlette==0.41.3",
    "sympy==1.13.1",
    "torch==2.5.1",
    "torchaudio==2.5.1",
    "torchvision==0.20.1",
    "triton==3.1.0",
    "typing_extensions==4.12.2",
    "urllib3==2.2.3",
    "uvicorn==0.34.0",
    )
)

app = modal.App("SAM", image=mask_image)

# Initialize FastAPI app
web_app = FastAPI()

# modal.Volume.from_name("root", create_if_missing=True)
# vol = modal.Volume.lookup("root")

# with vol.batch_upload() as batch:
#     batch.put_directory("/home/bilal/sahal/sam_serverless/segment_anything", "/root/segment_anything")

# Explicitly define paths
local_config_path = Path("/home/bilal/sahal/sam_serverless/config.py").resolve()
local_utils_path = Path("/home/bilal/sahal/sam_serverless/utils.py").resolve()
local_models_init_path = Path("/home/bilal/sahal/sam_serverless/models_init.py").resolve()
local_models_path = Path("/home/bilal/sahal/sam_serverless/models.py").resolve()
# local_SUPIR_path = Path("/home/nimra/sahal/SUPIR/SUPIR").resolve()
# local_sgm_path = Path("/home/nimra/sahal/SUPIR/sgm").resolve()

# Remote paths in the container
remote_config_path = Path("/root/config.py")
remote_utils_path = Path("/root/utils.py")
remote_models_init_path = Path("/root/models_init.py")
remote_models_path = Path("/root/models.py")
# remote_SUPIR_path = Path("/root/SUPIR")
# remote_sgm_path = Path("/root/sgm")

# Mount files to the container
mounts = [
    modal.Mount.from_local_file(local_models_init_path, remote_models_init_path),
    modal.Mount.from_local_file(local_utils_path, remote_utils_path),
    modal.Mount.from_local_file(local_config_path, remote_config_path),
    modal.Mount.from_local_file(local_models_path, remote_models_path),
    # modal.Mount.from_local_file(local_SUPIR_path, remote_SUPIR_path),
    # modal.Mount.from_local_file(local_sgm_path, remote_sgm_path),
]

@app.cls(
    gpu="t4",
    concurrency_limit=5,
    mounts=mounts,
    container_idle_timeout=120,  # in seconds
    volumes={
        "/root/sam_vit_h_4b8939": modal.Volume.from_name("sam_vit_h_4b8939", create_if_missing=True),
    },
)

class Model:
    @modal.enter()  #Enter the container
    def start_runtime(self):
        from models_init import initialize_models
        global model
        model = initialize_models()
        self.model = model
        print("Models initialized successfully")

    @modal.method()
    def segment_image(self,request: ImageRequest) -> ImageResponse:
        request_id = str(int(time.time()))
        start_time = time.time()
        logger.info(f"Request {request_id} received at {start_time}")
        item = request.input
        pil_image = base64_to_image(item.target_image)
        
        # Validate and reshape coordinates
        pos_coord = np.array(item.pos_coord)
        if pos_coord.ndim == 1:
            # Reshape flat list to list of lists
            pos_coord = pos_coord.reshape(-1, 2)
        elif pos_coord.ndim != 2 or pos_coord.shape[1] != 2:
            raise ValueError("pos_coord must be a list of [x, y] coordinate pairs.")
        
        masked_image = run_sam(pil_image, pos_coord, self.model)

        # Convert the image to base64 for the response
        img_base64 = image_to_base64(masked_image)
        return ImageResponse(output=Response(mask=f"data:image/png;base64,{img_base64}"))

# Instantiate the Modal Model class
app_model = Model()

# Define the endpoint
@web_app.post("/mask_image/", response_model=ImageResponse)
async def generate_images_endpoint(request: Request):
    try:
        # Parse the incoming JSON request into an ImageRequest object
        request_data = await request.json()
        image_request = ImageRequest(**request_data)

        # Call the generate_images method asynchronously
        image_response = await app_model.segment_image.remote.aio(image_request)
        return image_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve the FastAPI app using Modal
@app.function(
    image=mask_image,
    gpu="t4",
    concurrency_limit=5,
    mounts=mounts,
    container_idle_timeout=120,  # in seconds
    volumes={
        "/root/sam_vit_h_4b8939": modal.Volume.from_name("sam_vit_h_4b8939", create_if_missing=True),
    },
)

@modal.asgi_app()
def fastapi_app():
    return web_app