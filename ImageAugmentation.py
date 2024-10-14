import os
from PIL import Image

class ImageAugmentor:
    def __init__(self, source_folder, output_folder):
        self.source_folder = source_folder
        self.output_folder = output_folder
        
        # Create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def is_already_augmented(self, filename):
        """ Checks if an image has already been augmented by looking for rotated files """
        base_filename = os.path.splitext(filename)[0]
        for angle in [0, 90, 180, 270]:
            rotated_filename = f"{base_filename}_rotated_{angle}.jpg"
            if not os.path.exists(os.path.join(self.output_folder, rotated_filename)):
                return False
        return True

    def rotate_and_save(self, image_path, filename):
        """ Rotates the image by 0, 90, 180, and 270 degrees and saves them """
        img = Image.open(image_path)
        base_filename = os.path.splitext(filename)[0]

        for angle in [0, 90, 180, 270]:
            rotated_filename = f"{base_filename}_rotated_{angle}.jpg"
            rotated_filepath = os.path.join(self.output_folder, rotated_filename)
            
            # Skip if the rotated file already exists
            if os.path.exists(rotated_filepath):
                print(f"Skipping {rotated_filename}, already exists.")
                continue
            
            rotated_img = img.rotate(angle)
            rotated_img.save(rotated_filepath)
            # print(f"Saved {rotated_filename}")

    def process_images(self):
        """ Processes all .jpg images in the source folder """
        for filename in os.listdir(self.source_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.source_folder, filename)
                
                # Check if the image is already augmented
                if self.is_already_augmented(filename):
                    print(f"Skipping {filename}, already augmented.")
                    continue
                
                # Perform augmentation if not already augmented
                self.rotate_and_save(image_path, filename)