from ImageAugmentation import ImageAugmentor


class Main:
    @staticmethod
    def main():
        # Data Folders for cat and non-cat images
        image_folder_cat = "D:\\cat_data\\cat"
        image_folder_noncat = "D:\\cat_data\\noncat"

        # Output folders
        output_folder_cat = "D:\\cat_data\\cato"
        output_folder_noncat = "D:\\cat_data\\noncato"

        # Create instances of ImageAugmentor for cat and noncat folders
        cat_augmentor = ImageAugmentor(image_folder_cat, output_folder_cat)
        noncat_augmentor = ImageAugmentor(image_folder_noncat, output_folder_noncat)

        # Process images for both categories
        print("Processing cat images...")
        cat_augmentor.process_images()

        print("Processing noncat images...")
        noncat_augmentor.process_images()

        print(f"Rotated images saved to {output_folder_cat} & {output_folder_noncat}")


# Call the main method when running the script
if __name__ == "__main__":
    Main.main()