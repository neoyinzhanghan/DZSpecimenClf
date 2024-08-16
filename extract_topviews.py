import os
import openslide
from tqdm import tqdm
from PIL import Image

# Directory containing the .ndpi files
input_dir = '/media/hdd3/neo/BMA_AML'

# Function to process each .ndpi file
def process_ndpi_file(ndpi_file_path):
    try:
        # Open the ndpi file with OpenSlide
        slide = openslide.OpenSlide(ndpi_file_path)
        
        # Check if level 7 exists
        level = 7
        if level < slide.level_count:
            # Get the level 7 image
            img = slide.read_region((0, 0), level, slide.level_dimensions[level])
            img = img.convert("RGB")
            
            # Create output file path with .png extension
            output_file_path = ndpi_file_path.replace('.ndpi', '.png')
            
            # Save the image as .png
            img.save(output_file_path)
        else:
            print(f"Level 7 not available for {ndpi_file_path}")
        
        slide.close()
    
    except openslide.OpenSlideError as e:
        print(f"Error processing {ndpi_file_path}: {e}")

# Get all .ndpi files in the directory
ndpi_files = [f for f in os.listdir(input_dir) if f.endswith('.ndpi')]

# Process each file with progress tracking
for ndpi_file in tqdm(ndpi_files, desc="Processing NDPI files"):
    ndpi_file_path = os.path.join(input_dir, ndpi_file)
    process_ndpi_file(ndpi_file_path)

print("Processing completed.")
