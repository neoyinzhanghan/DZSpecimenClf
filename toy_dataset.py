import numpy as np
from PIL import Image

def create_rectangle(rectangle_size, class_id):
    img = np.zeros((rectangle_size, rectangle_size, 3), np.uint8)  # Create a small rectangle
    
    # Random color for the rectangle
    color = np.random.randint(0, 255, 3)
    img[:, :] = color
    
    if class_id == 0:
        # Add a small plus sign in the middle of the rectangle
        plus_size = 10
        center_x, center_y = rectangle_size // 2, rectangle_size // 2
        img[center_y - plus_size // 2 : center_y + plus_size // 2, center_x - 1 : center_x + 1] = [255, 255, 255]  # vertical line
        img[center_y - 1 : center_y + 1, center_x - plus_size // 2 : center_x + plus_size // 2] = [255, 255, 255]  # horizontal line
    
    elif class_id == 1:
        # Add a small circle in the middle of the rectangle
        circle_radius = 10
        center_x, center_y = rectangle_size // 2, rectangle_size // 2
        for i in range(rectangle_size):
            for j in range(rectangle_size):
                if (i - center_x) ** 2 + (j - center_y) ** 2 < circle_radius**2:
                    img[j, i] = [255, 255, 255]  # white circle
    
    else:
        raise ValueError("Invalid class_id")
    
    return img

# Parameters
rectangle_size = 224
downsample_rate = 16
width = 224 * 50
height = 224 * 20

# Create a blank image of size width x height
img = np.zeros((height, width, 3), np.uint8)

# Generate a rectangle with class_id = 0 or 1
rectangle = create_rectangle(rectangle_size, class_id=0)  # Change to class_id=1 for circle

# Randomly place the rectangle in the large image
x = np.random.randint(0, width - rectangle_size)
y = np.random.randint(0, height - rectangle_size)

# Place the rectangle in the main image
img[y : y + rectangle_size, x : x + rectangle_size] = rectangle

# Save the image
img = Image.fromarray(img)
img.save("rectangle.jpg")