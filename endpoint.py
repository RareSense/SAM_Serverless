from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models_init import initialize_models
from models import ImageRequest, ImageResponse
from fastapi import HTTPException
from config import logger, MAX_WORKERS, QUEUE_SIZE
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from utils import base64_to_image, image_to_base64,run_sam
import numpy as np

app = FastAPI(
    title="Segmentation API",
    description="API for Masking images using SAM",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = initialize_models()
logger.info("Models initialized successfully")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)  # Single worker to ensure sequential processing
request_queue = asyncio.Queue(maxsize=QUEUE_SIZE)  # Queue to hold incoming requests with a maximum size of 2

# Background worker to process the queue sequentially
async def process_queue():
    while True:
        request_id, request, future = await request_queue.get()
        try:
            # Process the request (offloaded to a separate thread if blocking)
            result = await asyncio.get_event_loop().run_in_executor(executor, segment_image, request)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()

# Start the background worker at startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

def segment_image(request: ImageRequest) -> ImageResponse:
    request_id = str(int(time.time()))
    start_time = time.time()
    logger.info(f"Request {request_id} received at {start_time}")

    pil_image = base64_to_image(request.target_image)
    
    # Validate and reshape coordinates
    pos_coord = np.array(request.pos_coord)
    if pos_coord.ndim == 1:
        # Reshape flat list to list of lists
        pos_coord = pos_coord.reshape(-1, 2)
    elif pos_coord.ndim != 2 or pos_coord.shape[1] != 2:
        raise ValueError("pos_coord must be a list of [x, y] coordinate pairs.")
    
    masked_image = run_sam(pil_image, pos_coord, model)

    # Convert the image to base64 for the response
    img_base64 = image_to_base64(masked_image)
    return {"mask": f"data:image/png;base64,{img_base64}"}

# def segment_image(request: Request) -> Response:
#     # Convert base64 to PIL image
#     request_id = str(int(time.time()))
#     start_time = time.time()
#     logger.info(f"Request {request_id} received at {start_time}")
    
#     pil_image = base64_to_image(request["target_image"])
#     pos_coord = np.array(request["pos_coord"])

#     masked_image = run_sam(pil_image,pos_coord,model)

#     #masked_image.save("masked_image.png")
#     # Convert the image to base64 for the response
#     img_base64 = image_to_base64(masked_image)
    
#     return {"mask": f"data:image/png;base64,{img_base64}"}

# Endpoint for image segmentation
@app.post("/mask_image/", response_model=ImageResponse)
async def generate_images(request: ImageRequest):
    request_id = str(int(time.time()))
    future = asyncio.get_event_loop().create_future()  # Create a future to hold the result

    try:
        # Attempt to add request to queue with a timeout
        await asyncio.wait_for(request_queue.put((request_id, request, future)), timeout=5.0)
    except asyncio.TimeoutError:
        # Return error if the queue is full and timeout is reached
        raise HTTPException(status_code=503, detail="Service is currently busy. Please try again later.")
    
    return await future  # Await the result from the queue processor

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "endpoint:app",
        host="0.0.0.0",
        port=8002,
        workers=1,  # Since using GPU, multiple workers might not be beneficial
        log_level="info"
    )  