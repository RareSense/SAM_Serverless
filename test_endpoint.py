import requests
import base64

# Function to encode an image to base64 (compatible with the server)
def load_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"

# Function to decode base64 to an image and save it
def save_base64_image(base64_string, output_path):
    try:
        # Remove prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode and save the image
        image_data = base64.b64decode(base64_string)
        with open(output_path, "wb") as output_file:
            output_file.write(image_data)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

# URL of the endpoint
url = "http://0.0.0.0:8000/mask_image/"

# Input data
input_image_path = "IMG-20241205-WA0018.jpg"  # Path to your input image
output_image_path = "masked_image.jpg"  # Path to save the masked image

# Convert the input image to base64
base64_image = load_image_to_base64(input_image_path)

# Define the payload with example coordinates
payload = {
    "target_image": base64_image,
    "pos_coord": [[100.0, 150.0], [200.0, 250.0]]  # Replace with your own coordinates
    
}

# Send the request to the API
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx

    # Parse the response JSON
    response_data = response.json()
    mask_base64 = response_data.get("mask")

    if mask_base64:
        # Save the masked image from base64
        save_base64_image(mask_base64, output_image_path)
    else:
        print("No 'mask' key found in the response.")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
