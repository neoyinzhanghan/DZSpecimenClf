import pandas as pd
import openslide
import time
from tqdm import tqdm

# Load the metadata CSV file
metadata_path = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"
df = pd.read_csv(metadata_path)

# Get the paths to the ndpi files
ndpi_paths = df['ndpi_path'].tolist()

# Number of repetitions per slide
n = 100

# Function to sample a point from level 0 and measure time
def sample_time(slide):
    width, height = slide.level_dimensions[0]
    start_time = time.time()
    _ = slide.read_region((width // 2, height // 2), 0, (1, 1))  # Sample the center pixel
    end_time = time.time()
    return end_time - start_time

# List to store average times per slide
average_times = []

# Iterate over each NDPI file path
for ndpi_path in tqdm(ndpi_paths, desc="Profiling slides"):
    slide = openslide.OpenSlide(ndpi_path)
    times = []
    for _ in range(n):
        times.append(sample_time(slide))
    average_time = sum(times) / len(times)
    average_times.append((ndpi_path, average_time))
    slide.close()

# Display the results
for ndpi_path, avg_time in average_times:
    print(f"Slide: {ndpi_path}, Average Sampling Time: {avg_time:.6f} seconds")
