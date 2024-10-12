import os
from PIL import Image

class ImageAugmentor:
    def __init__(self, source_folder, output_folder):
        self.source_folder = source_folder
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def rotate_and_save(self, image_path, filename):
        """ Rotates the image by 90, 180, and 270 degrees and saves them """
        img = Image.open(image_path)
        for angle in [0, 90, 180, 270]:
            rotated_img = img.rotate(angle)
            new_filename = f"{os.path.splitext(filename)[0]}_rotated_{angle}.jpg"
            rotated_img.save(os.path.join(self.output_folder, new_filename))

    def process_images(self):
        """ Processes all .jpg images in the source folder """
        for filename in os.listdir(self.source_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.source_folder, filename)
                self.rotate_and_save(image_path, filename)