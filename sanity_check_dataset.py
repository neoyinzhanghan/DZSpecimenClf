import os
import pandas as pd
import openslide
from tqdm import tqdm

metadata_file = "/home/greg/Documents/neo/wsi_specimen_clf_metadata.csv"

# open metadata file
df = pd.read_csv(metadata_file)

bad_files = []

# get the list of the ndpi paths
ndpi_paths = df["ndpi_path"].tolist()

# check if the ndpi files are valids
for ndpi_path in tqdm(ndpi_paths, desc="Checking NDPI files"):
    try:
        slide = openslide.OpenSlide(ndpi_path)
        # and try to extract the level 7 image
        level = 7
        if level < slide.level_count:
            img = slide.read_region((0, 0), level, slide.level_dimensions[level])
            img = img.convert("RGB")
        else:
            print(f"Level 7 not available for {ndpi_path}")
        slide.close()
    except openslide.OpenSlideError as e:
        print(f"Error processing {ndpi_path}: {e}")

        # add the bad file to the list
        bad_files.append(ndpi_path)

print(f"Bad files: {bad_files}")
