
![final2](https://github.com/user-attachments/assets/ad3461d4-1af6-4d4f-99e5-915f51c6b5d0)


# Cat Identification Using Deep Learning

Cat Detector is a deep learning project designed to determine whether a cat is present in a given image. Focused on enhancing the programmer's understanding of deep learning, **this project is developed without relying on pre-built libraries like TensorFlow**. It solely revolves around the model, with no backend or frontend integration. After training the model on images, users can test their own images to detect the presence of a cat.

## Features

- **Cat Detection:** Identifies if a cat is present in any given image.
- **Simple Image Processing:** Uses a trained deep learning model to make predictions on images.
- **No Backend/Frontend:** A lightweight solution focused solely on the deep learning model and image testing.


## Tech Stack

- **Programming Language:** Python
- **Python Libraries**
    - NumPy: For matrix operations and numerical computations.
    - Image (Pillow): For image processing and handling.


## Installation

- Install my-project with npm

```bash
git clone https://github.com/yourusername/cat-detector.git
cd cat-detector
```
- Install required libraries:
```bash
pip install numpy Image
```


## Purpose and Logic of the Project



- Purpose: To develop a project without using pre-built libraries like TensorFlow, in order to enhance the programmer's understanding of deep learning.

![paintsss](https://github.com/user-attachments/assets/e21ec42d-e0a2-4ba0-b05d-e2d6a0160605)
- Logic: 
    - An image is transformed into a matrix by separating it into Red, Green, and Blue (RGB) layers.
    - For example, if an image has a size of 64x64 pixels, it generates three 64x64 matrices—one for each color channel (R, G, B).
    - These three matrices are then concatenated into a single 2D matrix with a shape of (64x64x3, 1), which simplifies to (12288, 1).
    - This 2D matrix represents the entire image and is denoted as `x(m)`, where `m` refers to the `m`-th example in the dataset. 
    ![rgb_mat_transparent](https://github.com/user-attachments/assets/c265adde-2589-4aca-8ad2-6adb527ad6b2)    

    - The concatenation of x arrays into a single array, denoted as X, represents the process of **vectorization**. Vectorization provides faster training compared to an explicit for loop.![rgb_final](https://github.com/user-attachments/assets/8bd90896-283b-4e07-b2bb-4b374ee8c4c6)






