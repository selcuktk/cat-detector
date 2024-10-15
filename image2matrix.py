import os
from PIL import Image
import numpy as np

class i2m:
    # This function, create individual an xm array from given specific image
    @staticmethod
    def process_image(image_path):
        # Load the image
        img = Image.open(image_path)
        img_array = np.array(img)  # Convert the image to a NumPy array

        # Separate the Red, Green, and Blue channels
        red_channel = img_array[:, :, 0]  # Red channel
        green_channel = img_array[:, :, 1]  # Green channel
        blue_channel = img_array[:, :, 2]  # Blue channel

        # Reshape channels to 2D column vectors
        red_channel = red_channel.reshape(red_channel.size, 1)
        green_channel = green_channel.reshape(green_channel.size, 1)
        blue_channel = blue_channel.reshape(blue_channel.size, 1)
        
        # Concatenate the three channels into a single matrix
        xm = np.concatenate((red_channel, green_channel, blue_channel), axis=0)

        return xm


    # This method processes all images in the folder, converting each into a NumPy array (xm).
    # All individual xm matrices are then concatenated to form a larger matrix (Xm).
    @staticmethod
    def process_folder(folder_path):
        xm_list = [] # That is a list that is going to contain the matrix of every .jpg file in a folder. 
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                xm = i2m.process_image(image_path)
                xm_list.append(xm)   # Append the processed matrix to the list
        
        # Concatenate all matrices in the list into a single matrix
        if xm_list:
            Xm = np.concatenate(xm_list, axis=1)
            print(f"Combined matrix shape: {Xm.shape}")
        else:
            print("No images found or processed.")

        return Xm