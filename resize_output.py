import os
import cv2

# Set folder paths (use raw strings to avoid escape character issues)
folder_path = r'D:\python-learn\K\seg-MOGA\outputs\lizi_UKAN\out_val'  # Path to predicted result masks
input_folder_path = r'D:\python-learn\K\seg-MOGA\inputs\lizi\images' # Path to input images
save_folder_path = r'D:\python-learn\K\seg-MOGA\UKAN\out_val_resiz' # Resized results output path

# Check if directories exist
print(f"Checking directories...")
print(f"Prediction results directory: {folder_path} - {'Exists' if os.path.exists(folder_path) else 'Does not exist'}")
print(f"Input images directory: {input_folder_path} - {'Exists' if os.path.exists(input_folder_path) else 'Does not exist'}")

# Create output directory
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print(f"Created output directory: {save_folder_path}")

# Check if prediction results directory exists
if not os.path.exists(folder_path):
    print(f"Prediction results directory does not exist: {folder_path}")
    exit(1)

# Get paths of all image files
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_paths)} prediction result files")

# Read and process each image
processed_count = 0
error_count = 0

for image_path in image_paths:
    name = os.path.basename(image_path).replace('.jpg', '.png')
    input_path = os.path.join(input_folder_path, name)
    
    try:
        # Read original image and predicted mask
        image = cv2.imread(input_path)
        mask = cv2.imread(image_path, 0)
        
        # Check if images were read successfully
        if image is None:
            print(f"Failed to read input image: {name}")
            error_count += 1
            continue
            
        if mask is None:
            print(f"Failed to read predicted mask: {name}")
            error_count += 1
            continue
        
        # Get dimensions of the input image
        base_height, base_width = image.shape[:2]

        # Set target dimensions for resizing (matching base image dimensions)
        new_width = base_width
        new_height = base_height

        # Resize mask using cv2.resize()
        resized_mask = cv2.resize(mask, (new_width, new_height))
        
        # Save resized mask
        output_path = os.path.join(save_folder_path, name)
        cv2.imwrite(output_path, resized_mask)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} files...")
            
    except Exception as e:
        print(f"Error processing file {name}: {e}")
        error_count += 1

print(f"\nProcessing complete:")
print(f"   Successfully processed: {processed_count} files")
print(f"   Files with errors: {error_count} files")
print(f"   Output directory: {save_folder_path}")
