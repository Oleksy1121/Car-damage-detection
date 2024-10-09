# YOLO Object Detection Image Viewer

## Description
The application is designed for detecting car body damages using the YOLO (You Only Look Once) model. It enables users to load a folder containing images, perform damage detection, and visualize the results as annotated images. The application features a graphical user interface (GUI) built with the Tkinter library, allowing seamless navigation through images and detailed review of detection results.

## Features
- **Load Images**: Easily load images from a selected folder for processing.
- **Car Body Damage Detection**: Utilize a YOLO model to detect various types of car body damage.
- **Annotated Results**: Display images with highlighted areas indicating detected damages.
- **Image Navigation**: Navigate through images using "Previous" and "Next" buttons.
- **Detailed Logging**: A log window presents detailed information about detected damages.
- **Drag-and-Drop Support**: Conveniently drag and drop folders directly into the application.
- **Responsive Image Display**: Automatically adjusts the displayed image to fit the window size.

## Sample Application Screenshot
Below is a sample screenshot showcasing the application's functionality:

![Sample Application](../img/app_screen.png)

## Running the Application
It is recommended to open the application's folder as a project in a development environment such as PyCharm. Ensure that all required dependencies are installed.

1. **Open the Project**: In PyCharm, select the application's main folder as the project directory.
2. **Install Dependencies**: Run the following command in the terminal to install all required packages:
   ```bash
   pip install -r requirements.txt
3. **Run the Application**: Execute the following command to launch the application:
   ```bash
   python app.py
