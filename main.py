from ImageAugmentation import ImageAugmentor


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
        

        




if __name__ == "__main__":
    Main.main()