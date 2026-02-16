from PIL import Image
import os
import glob

# Set input and output folder paths
input_folder = "/mnt/bn/bruceyanglq/code/sphinx/seg/inputs/lizi/masks/0"  # Replace with your input folder path
output_folder = "vis"  # Replace with your output folder path

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Supported image formats
extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')

# Iterate through all image files
for ext in extensions:
    for img_path in glob.glob(os.path.join(input_folder, ext)):
        try:
            with Image.open(img_path) as img:
                # Convert to grayscale image
                gray_img = img.convert('L')
                # Apply binarization: set pixel values > 1 to 255, otherwise 0
                binary_img = gray_img.point(lambda x: 255 if x > 1 else 0)
                # Build output path
                filename = os.path.basename(img_path)
                output_path = os.path.join(output_folder, filename)
                # Save processed image
                binary_img.save(output_path)
                print(f"Processed successfully: {filename}")
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
