import os
import openslide
from tqdm import tqdm
from PIL import Image
import ray

# Initialize Ray
ray.init()

# Directory containing the .ndpi files
input_dir = '/media/hdd3/neo/PBS_slides'

# Define an actor class to process .ndpi files
@ray.remote
class NDPIProcessor:
    def process_file(self, ndpi_file_path):
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

# Create a pool of 5 actors
actors = [NDPIProcessor.remote() for _ in range(5)]

# Function to distribute tasks to the actor pool
def distribute_tasks(ndpi_file):
    actor = actors.pop(0)
    result = actor.process_file.remote(ndpi_file)
    actors.append(actor)
    return result

# Use tqdm to track the progress
results = []
with tqdm(total=len(ndpi_files), desc="Processing NDPI files") as pbar:
    for ndpi_file in ndpi_files:
        ndpi_file_path = os.path.join(input_dir, ndpi_file)
        results.append(distribute_tasks(ndpi_file_path))
        
        # Update the progress bar after each task completes
        completed_tasks = ray.get(results)
        pbar.update(len(completed_tasks))
        results.clear()  # Clear the results to ensure each update is only for the latest batch

print("Processing completed.")

# Shut down Ray
ray.shutdown()
