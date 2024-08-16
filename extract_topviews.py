import os
import openslide
from tqdm import tqdm
from PIL import Image
import ray

# Initialize Ray
ray.init()

# Directory containing the .ndpi files
input_dir = '/media/hdd3/neo/BMA_AML'

# Remote function to process each .ndpi file in parallel
@ray.remote
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
            
            # Create output file path with .jpg extension
            output_file_path = ndpi_file_path.replace('.ndpi', '.jpg')
            
            # Save the image as .jpg with quality 95
            img.save(output_file_path, 'JPEG', quality=95)
        else:
            print(f"Level 7 not available for {ndpi_file_path}")
        
        slide.close()
    
    except openslide.OpenSlideError as e:
        print(f"Error processing {ndpi_file_path}: {e}")

# Get all .ndpi files in the directory
ndpi_files = [f for f in os.listdir(input_dir) if f.endswith('.ndpi')]

# Create a list of tasks
tasks = [process_ndpi_file.remote(os.path.join(input_dir, ndpi_file)) for ndpi_file in ndpi_files]

# Use tqdm to track the progress
for _ in tqdm(ray.get(tasks), desc="Processing NDPI files"):
    pass

print("Processing completed.")

# Shut down Ray
ray.shutdown()
