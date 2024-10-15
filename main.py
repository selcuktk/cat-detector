from ImageAugmentation import ImageAugmentor
from image2matrix import i2m
import numpy as np

class Main:
    @staticmethod
    def main():

        # Augmentation of .jpg files. 
        # All .jpg files are augmented by rotating 0, 90, 180, 270 degrees and put into an output folder

        # Data folder paths for cat and noncat images
        image_folder_cat = "raw-data\\cat"
        image_folder_noncat = "raw-data\\noncat"

        # Output folders
        output_folder_cat = "augmented-data\\augmented_cat"
        output_folder_noncat = "augmented-data\\augmented_noncat"

        # Create instances of ImageAugmentor for cat and noncat folders
        cat_augmentor = ImageAugmentor(image_folder_cat, output_folder_cat)
        noncat_augmentor = ImageAugmentor(image_folder_noncat, output_folder_noncat)

        # Process images for both categories
        print("Processing cat images...")
        cat_augmentor.process_images()
        print("Processing noncat images...")
        noncat_augmentor.process_images()
        print(f"Rotated images saved to {output_folder_cat} & {output_folder_noncat}")
        

        Xm_cat = i2m.process_folder(output_folder_cat)  
        Xm_noncat =  i2m.process_folder(output_folder_noncat)

        Xm = np.concatenate([Xm_cat, Xm_noncat], axis=1)
        print(Xm)
        print()
        print(Xm.shape)

        Ym_cat = np.ones(252).reshape(1, 252)
        Ym_noncat = np.zeros(248).reshape(1, 248)
        Ym = np.concatenate([Ym_cat, Ym_noncat], axis=1)

        print(Ym_cat.shape)
        print(Ym_noncat.shape)
        print(Ym.shape)

        print(Ym_cat)
        print(Ym_noncat)
        print(Ym)
        

        




if __name__ == "__main__":
    Main.main()