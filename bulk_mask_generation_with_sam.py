import requests
import base64
import json
import os
from PIL import Image  # <-- Make sure Pillow is installed

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
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def merge_images_horizontally(image_path_1, image_path_2, output_path):
    """
    Open two images and merge them side by side (left-to-right).
    Saves the result at 'output_path'.
    """
    try:
        img1 = Image.open(image_path_1)
        img2 = Image.open(image_path_2)

        # Determine final dimensions
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)

        # Create a new blank image with the combined width and max height
        new_img = Image.new("RGB", (total_width, max_height))

        # Paste img1 at (0,0)
        new_img.paste(img1, (0, 0))
        # Paste img2 immediately after img1 in the x-axis
        new_img.paste(img2, (img1.width, 0))

        # Save the merged image
        new_img.save(output_path)
        print(f"Merged image saved to {output_path}")

    except Exception as e:
        print(f"Error merging images: {e}")

def process_images(folder_path, json_file_path, endpoint_url):
    """
    Reads from the JSON file line-by-line, skips images with empty coordinates,
    sends a request to the endpoint for each valid entry, saves the mask,
    and merges the original image with the mask horizontally for comparison.
    """

    if not os.path.isfile(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found.")
        return

    with open(json_file_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            print(f"Skipping line {idx}: Invalid JSON format.")
            continue

        target = data.get("target")
        coordinates = data.get("coordinates", [])

        # Skip if coordinates are empty
        if not coordinates:
            print(f"Skipping '{target}' (line {idx}) because coordinates are empty.")
            continue

        # Build the local path to the image
        image_path = os.path.join(folder_path, target)
        if not os.path.exists(image_path):
            print(f"Skipping '{target}' (line {idx}) because file not found: {image_path}")
            continue

        # Convert image to Base64
        base64_image = load_image_to_base64(image_path)

        # Prepare the payload
        payload = {
            "input": {
                "target_image": base64_image,
                "pos_coord": coordinates
            }
        }

        print(f"Sending request to endpoint for image '{target}' (line {idx})...")

        try:
            response = requests.post(endpoint_url, json=payload)
            response.raise_for_status()

            response_data = response.json()
            print("Response received from server:")
            print(json.dumps(response_data, indent=4))

            # Get mask from response
            mask_base64 = response_data.get("output", {}).get("mask")
            folder_name = response_data.get("output", {}).get("folder_name", "default_masks")

            if mask_base64:
                # Paths to save the mask and the merged image
                mask_filename = f"masked_{os.path.splitext(target)[0]}.png"
                mask_output_path = os.path.join(folder_name, mask_filename)

                # Save mask image
                save_base64_image(mask_base64, mask_output_path)

                # Create a name for the merged image
                merged_filename = f"comparison_{os.path.splitext(target)[0]}.png"
                merged_output_path = os.path.join(folder_name, merged_filename)

                # Merge original + mask horizontally
                merge_images_horizontally(image_path, mask_output_path, merged_output_path)

            else:
                print("Warning: 'mask' key not found in response.")

        except requests.exceptions.RequestException as e:
            print(f"Request failed for image '{target}': {e}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response for image '{target}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred for image '{target}': {e}")

def main():
    folder_path = "/home/nimra/segmentation_background_mask_checker/Not_in_training_samples_11300"
    json_file_path = "/home/nimra/coordinates_generator/new_coordinate_generate/cords_set1_result.json"
    endpoint_url = "https://nemoooooooooo--sam-fastapi-app.modal.run/mask_image/"

    process_images(folder_path, json_file_path, endpoint_url)

if __name__ == "__main__":
    main()