import pickle
import numpy as np
from image2matrix import i2m
from dl_utils import dlf


class Main:
    @staticmethod
    def main():    # Read from file
        final_parameters = {}
        with open("final_parameters.pkl", "rb") as file:
            final_parameters = pickle.load(file)

        image_folder_cat = "test-data\\test-cat"
        image_folder_noncat = "test-data\\test-noncat"

        X_test_cat = i2m.process_folder(image_folder_cat)
        X_test_noncat = i2m.process_folder(image_folder_noncat)

        X_test = np.concatenate([X_test_cat, X_test_noncat], axis=1)
        X_test = X_test/255

        print(X_test.shape)
        Y_test_cat = np.ones(50).reshape(1, 50)
        Y_test_noncat = np.zeros(50).reshape(1, 50)
        Y_test = np.concatenate([Y_test_cat, Y_test_noncat], axis=1)


        AL, stores = dlf.L_model_forward(X_test, final_parameters)
        cost = dlf.compute_cost(AL, Y_test)

        print("The cost is: " + str(cost))
        print(final_parameters.keys())
        print(final_parameters.get("W1").shape)


if __name__ == "__main__":
    Main.main()