from pydantic import BaseModel, Field, validator
from typing import List
import base64


class Request(BaseModel):
    target_image: str = Field(..., description="Base64 encoded string of the target image to be segmented.")
    pos_coord: List[List[float]] = Field(..., description="Array of positive coordinates for the segmentation points.")
    
    @validator("target_image")
    def validate_base64_image(cls, value):
        """
        Validates that the `target_image` is a valid base64 string.
        """
        try:
            # Check if the string starts with 'data:image/' and contains a valid base64 image
            if not value.startswith("data:image/"):
                raise ValueError("Image must start with 'data:image/'.")
            header, encoded = value.split(",", 1)
            base64.b64decode(encoded)
        except Exception as e:
            raise ValueError("Invalid base64 image format.") from e
        return value
    
    @validator("pos_coord")
    def validate_coordinates(cls, value):
        """
        Validates that the `pos_coord` list is not empty and contains valid float numbers.
        """
        if not value or len(value) == 0:
            raise ValueError("pos_coord must contain at least one coordinate.")
        return value


class Response(BaseModel):
    mask: str = Field(..., description="Base64 encoded string of the masked image.")

    @validator("mask")
    def validate_base64_mask(cls, value):
        """
        Validates that the `mask` is a valid base64 string with the appropriate prefix.
        """
        try:
            # Check if the string starts with 'data:image/' and contains a valid base64 image
            if not value.startswith("data:image/png;base64,"):
                raise ValueError("Masked image must start with 'data:image/png;base64,'.")
            header, encoded = value.split(",", 1)
            base64.b64decode(encoded)
        except Exception as e:
            raise ValueError("Invalid base64 mask format.") from e
        return value

class ImageRequest(BaseModel):
    input: Request

class ImageResponse(BaseModel):
    output: Response
