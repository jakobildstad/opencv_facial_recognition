from PIL import Image
import os

# Define the folder containing the images
folder_path = "/Users/jakobildstad/Documents/Projects/opencv_facial_recognition_private/raw_training_data/jakob"

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Process only PNG and DNG files (case insensitive)
    if filename.lower().endswith('.dng'): #or filename.lower().endswith('.png'):
        input_path = os.path.join(folder_path, filename)
        
        # Determine output path
        if filename.lower().endswith('.png'):
            output_path = input_path  # Overwrite the original file
        else:  # for .dng files
            base, _ = os.path.splitext(input_path)
            output_path = base + '.png'

        try:
            with Image.open(input_path) as img:
                # Convert to a standard RGB mode to remove any unusual formatting
                img.convert('RGB').save(output_path, 'PNG')
                print(f"Processed {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")