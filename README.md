# Car Damage Detection

This project focuses on detecting and localizing various types of car body damages using deep learning models. It includes a custom YOLOv8 model and a Faster R-CNN model, along with a user-friendly application for image processing and visualization.

## Dataset
The dataset used for training contains images of car body damages, including classes such as scratches, rust, paint fading, paint cracks, dents, cracks, and PDR dents. It was created using the Roboflow platform and is available in two versions:
- **Without Data Augmentation**: 456 images.
- **With Data Augmentation**: 1140 images.

All images have been annotated using polygons to ensure high precision in labeling, which was a time-consuming but crucial process due to the relatively small dataset size. This precise annotation was necessary for effective model training.

You can access the dataset and more details on Roboflow here: [Roboflow Project Link](https://universe.roboflow.com/cardetecion/car-paint-damage-detection)

## Project Structure
The repository is organized as follows:
```
├── damage_detection_app/
│   ├── app.py               # Main script for the YOLO-based application.
│   ├── requirements.txt     # Required dependencies.
│   ├── README.md            # Instructions specific to the application.
│   └── test/                # Folder containing test images.
├── img/                     # Contains sample images and screenshots.
│   └── app_screen.png       # Example screenshot of the application in use.
├── training/
│   ├── YOLOv8.ipynb         # Jupyter notebook for training the YOLOv8 model.
│   └── Faster R-CNN.ipynb   # Jupyter notebook for training the Faster R-CNN model.
└── README.md                # Main instructions and overview of the project.
```

## Features
- **Dataset Creation**: High-quality image dataset with precise annotations using polygons.
- **Model Training**: Training of YOLOv8 and Faster R-CNN models using Google Colab.
- **Model Comparison**: Both models were trained and compared for their performance in detecting car body damages.
- **Application**: A Python application for detecting damages in images, with a graphical user interface (GUI) for easy navigation and result visualization.

## How to Use
1. **Download or clone the repository**:
   ```bash
   git clone https://github.com/Oleksy1121/Car-Damage-Detection.git
   cd Car-Damage-Detection
   ```

2. **Navigate to the `damage_detection_app/` folder** and follow the instructions in the `README.md` for setting up and running the application.

3. **To train models**, open the Jupyter notebooks in the `training/` folder using Google Colab or another environment.
   - For training the YOLOv8 model, use the `YOLOv8.ipynb` notebook.
   - For training the Faster R-CNN model, use the `Faster R-CNN.ipynb` notebook.
   - Each notebook includes detailed steps for setting up the dataset, configuring the model, and training.

4. **Install Dependencies**: Run the following command in the terminal to install all required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Application**: Execute the following command to launch the application:
    ```
    python app.py
    ```