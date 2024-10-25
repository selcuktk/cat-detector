from ImageAugmentation import ImageAugmentor
from image2matrix import i2m
from dl_utils import activation
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

        # Augmentation of images for both categories
        print("Processing cat images...")
        cat_augmentor.process_images()
        print("Processing noncat images...")
        noncat_augmentor.process_images()
        print(f"Rotated images saved to {output_folder_cat} & {output_folder_noncat}")
        

        # Creating Xm matrix from data
        Xm_cat = i2m.process_folder(output_folder_cat)  
        Xm_noncat =  i2m.process_folder(output_folder_noncat)
        Xm = np.concatenate([Xm_cat, Xm_noncat], axis=1)

        Xm = Xm/255


        print(Xm.shape)
        print("--------------")


        # Creating Ym matrix manually
        Ym_cat = np.ones(252).reshape(1, 252)
        Ym_noncat = np.zeros(248).reshape(1, 248)
        Ym = np.concatenate([Ym_cat, Ym_noncat], axis=1)

        Xm, Ym = i2m.shuffle_xy_together(Xm, Ym)


        print(Ym_cat.shape)
        print(Ym_noncat.shape)
        print(Ym.shape)
        print("--------------")

        wL1 = np.random.randn(6, 12288) * 0.01
        wL2 = np.random.randn(5, 6) * 0.01
        wL3 = np.random.randn(1, 5) * 0.01

        """
        9997     0.6928421656779147
        9998     0.6928416569351841
        9999     0.6928411469188254
        """
        
        bL1 = np.zeros((6, 1))
        bL2 = np.zeros((5, 1))
        bL3 = np.zeros((1, 1))

        J = 0

        learning_rate = 0.05
        num_iterations = 10000


        for i in range (num_iterations):
                m = Xm.shape[1]  # number of examples
                J = 0

                # Forward propagation
                ZL1 = np.dot((wL1), Xm) + bL1
                AL1 = activation.sigmoid(ZL1)

                ZL2 = np.dot((wL2), AL1) + bL2
                AL2 = activation.sigmoid(ZL2)

                ZL3 = np.dot((wL3), AL2) + bL3
                AL3 = activation.sigmoid(ZL3)
                
                # Cost to print
                J += -((Ym*(np.log(AL3))) + (1-Ym)*(np.log(1-AL3)))
                J = np.sum(J)/m

                # Back propagation
                dzL3 = AL3 - Ym
                dwL3 = (1/m) * np.dot(dzL3, AL2.T)
                dbL3 = (1/m) * np.sum(dzL3, axis=1, keepdims = True)

                dzL2 = np.dot(wL3.T, dzL3) * activation.sigmoid_derivative(AL2)
                dwL2 = (1/m) * np.dot(dzL2, AL1.T)
                dbL2 = (1/m) * np.sum(dzL2, axis=1, keepdims=True)

                dzL1 = np.dot(wL2.T, dzL2) * activation.sigmoid_derivative(AL1)
                dwL1 = (1/m) * np.dot(dzL1, Xm.T)
                dbL1 = (1/m) * np.sum(dzL1, axis=1, keepdims=True)

                # Gradient Descent - Update weights and biases
                wL1 -= learning_rate * dwL1
                bL1 -= learning_rate * dbL1

                wL2 -= learning_rate * dwL2
                bL2 -= learning_rate * dbL2

                wL3 -= learning_rate * dwL3
                bL3 -= learning_rate * dbL3
                
                print(f"{i} \t {J}")
        
        
            


if __name__ == "__main__":
    Main.main()