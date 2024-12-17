import requests
import base64
import json
from PIL import Image
from io import BytesIO
import os

def load_image_to_base64(image_path):
    """Encode an image file to base64 format with the proper prefix."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"

def save_base64_image(base64_string, output_path):
    """Decode a base64 image and save it as a file."""
    try:
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]
        image_data = base64.b64decode(base64_string)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def save_schema_to_file(payload, file_path):
    """Save the payload schema to a text file."""
    try:
        with open(file_path, "w") as file:
            file.write(json.dumps(payload, indent=4))
        print(f"Schema saved to {file_path}")
    except Exception as e:
        print(f"Error saving schema to file: {e}")

def test_endpoint():
    """Main function to test the mask image API endpoint."""
    # Endpoint URL
    url = "https://nemoooooooooo--sam-fastapi-app.modal.run/mask_image/"  # Replace with the correct server URL if needed

    # Image paths
    input_image_path = "IMG-20241205-WA0018.jpg"  # Replace with a valid local image path
    output_image_path = "masked_output.png"       # File to save the masked image response
    schema_output_path = "payload_schema.txt"     # File to save the payload schema

    if not os.path.exists(input_image_path):
        print(f"Error: Input image file '{input_image_path}' not found.")
        return

    # Step 1: Convert input image to base64
    base64_image = load_image_to_base64(input_image_path)
    
    # Step 2: Prepare the payload
    payload = {
        "input": {
        "target_image": base64_image,
        "pos_coord": [[700, 450], [700, 550]]
        }  # Example coordinates, replace as needed
    }

    # Save the payload schema to a file
    save_schema_to_file(payload, schema_output_path)

    print("Sending request to endpoint...")
    try:
        # Step 3: Send POST request to the server
        response = requests.post(url, json=payload)
        response.raise_for_status()

        # Step 4: Parse response JSON
        response_data = response.json()
        print("Response received from server:")
        print(json.dumps(response_data, indent=4))

        # Step 5: Validate and save masked image
        mask_base64 = response_data.get("output", {}).get("mask")
        if mask_base64:
            save_base64_image(mask_base64, output_image_path)
        else:
            print("Error: 'mask' key not found in response.")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_endpoint()
