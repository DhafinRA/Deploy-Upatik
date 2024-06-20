from io import BytesIO
from PIL import Image
import numpy as np
import base64
import csv
import cv2

# # Function to read label mappings from CSV file
# def read_label_mappings_from_csv(csv_file_path):
#     label_mapping = {}
#     with open(csv_file_path, mode='r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         for row in csv_reader:
#             index = int(row['Index'])
#             name = row['Name']
#             calories = float(row['Calories'])
#             fiber = float(row['Fiber'])
#             sugar = float(row['Sugar'])
#             label_mapping[index] = {"name": name, "nutrition": {"calories": calories, "fiber": fiber, "sugar": sugar}}
#     return label_mapping

# Function to read food data from CSV file
def read_label_mappings_from_csv(csv_file_path):
    label_mapping = []
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            food_id = row['id']
            food_name = row['Nama Makanan']
            calories = float(row['Kalori'])
            fat = float(row['Lemak'].replace('g', '').replace(',', '.'))
            carbohydrate = float(row['Karbohidrat'].replace('g', '').replace(',', '.'))
            protein = float(row['Protein'].replace('g', '').replace(',', '.'))
            category = row['Kategori']
            label_mapping.append({
                'id': food_id,
                'name': food_name,
                'calories': calories,
                'fat': fat,
                'carbohydrate': carbohydrate,
                'protein': protein,
                'category': category
            })
    return label_mapping

def image_to_base64(image):
    # Load and preprocess the image
    image = Image.open(BytesIO(image))
    image = image.resize((224, 224))
    image = image.convert('RGB')  # Resize if necessary
    image_array = np.array(image)

    # Normalize the pixel values to the range [0, 1]
    image_array = image_array / 255.0

    # If your model expects a specific input shape, adjust accordingly
    # For example, a common shape is (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    # Convert to list (or use base64 if preferred)
    image_list = image_array.tolist()

    # Alternatively, you can use base64 encoding
    _, buffer = cv2.imencode('.jpg', (image_array[0] * 255).astype(np.uint8))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Create the payload
    payload = {
        "instances": image_list  # or [{"b64": image_base64}]
    }

    return payload