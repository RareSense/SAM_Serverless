import base64
import io
from PIL import Image
from io import BytesIO
import random
from PIL import Image, ImageDraw
import numpy as np

def image_to_base64(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")  # Adjust format as needed
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def base64_to_image(base64_string):
    try:
        # Check and remove the prefix if it's there
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode the base64 string
        try:
            image_data = base64.b64decode(base64_string)
        except base64.binascii.Error as e:
            print(f"Error decoding base64 string: {e}")
            return None
        
        # Load the image data into a PIL Image object
        try:
            image = Image.open(BytesIO(image_data))
            # Convert the image to RGB
            rgb_image = image.convert('RGB')
            return rgb_image
        except IOError as e:
            print(f"Error loading image data: {e}")
            return None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def run_sam(image_pil, coordinates, sam_predictor):
    
    size = image_pil.size

    point_coords = coordinates
    point_labels = np.ones(point_coords.shape[0])

    image = np.array(image_pil)
    sam_predictor.set_image(image)

    print(f"shape of image: {image.shape}")

    masks, _, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=False,
    )

    mask_image = Image.new('RGB', size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask, mask_draw, random_color=False)

    mask_image.save("masked.png")
    return mask_image

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255), 153)
    else:
        # color = (30, 144, 255, 153)
        color = (255, 255, 255, 255)
        # color = (245, 165, 0, 0)


    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)