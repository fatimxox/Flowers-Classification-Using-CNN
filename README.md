
## Prerequisites

To run this project locally, you need to have Python installed on your system.

*   **Python:** Version 3.7 or higher is recommended.
*   **Python Libraries:** Install the required libraries using pip. It's highly recommended to use a virtual environment.
    ```bash
    pip install Flask tensorflow tensorflow-keras numpy opencv-python pillow matplotlib seaborn scikit-learn wordcloud imageio
    ```
    *   `Flask`: To build the web application.
    *   `tensorflow`, `tensorflow-keras`: For building and loading the deep learning model.
    *   `numpy`: For numerical operations.
    *   `opencv-python` (cv2): For image reading and preprocessing (e.g., resizing, color conversion).
    *   `Pillow`: For image handling (used by ImageDataGenerator and others).
    *   `matplotlib`, `seaborn`, `wordcloud`, `imageio`: For data exploration and visualization in the notebook.
    *   `scikit-learn`: For data splitting and evaluation metrics in the notebook.
*   **Dataset:** The `flower_photos` dataset, organized into subdirectories for each class (daisy, dandelion, roses, sunflowers, tulips), is required. It should be placed in the project's root directory. This dataset is commonly available on platforms like Kaggle or TensorFlow datasets.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatimxox/Flowers-Classification-Using-CNN.git
    cd Flowers-Classification-Using-CNN
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:** With your virtual environment activated, install the required libraries:
    ```bash
    pip install -r requirements.txt # If you create a requirements.txt
    # OR manually install if you don't have requirements.txt
    pip install Flask tensorflow tensorflow-keras numpy opencv-python pillow matplotlib seaborn scikit-learn wordcloud imageio
    ```
    *(You can generate a `requirements.txt` file after installing dependencies using `pip freeze > requirements.txt`)*

4.  **Obtain the dataset:** Download the `flower_photos` dataset and ensure the `flower_photos` directory with its subfolders is placed in the project's root directory.
5.  **Run the Notebook:** Execute all cells in `Flowers-Classification-Using-CNN copy.ipynb`. This notebook downloads pre-trained weights (requires internet access), preprocesses the data, trains the EfficientNetB3 model using transfer learning, and saves the trained model as `model.h5`. **Ensure the `model.h5` file is saved in the same directory as `app.py`**.

## How it Works

1.  **Model Training (Notebook):** The Jupyter Notebook (`Flowers-Classification-Using-CNN copy.ipynb`) details the model development process. It loads the dataset, visualizes sample images, performs basic data exploration (checking corrupted images, size distribution, category distribution), splits the data, creates `ImageDataGenerator` objects for training/validation/testing with augmentation for the training set, loads `EfficientNetB3` with pre-trained weights (excluding the top classification layer), adds custom layers (Batch Normalization, Dense, Dropout, Output Dense with Softmax) for the 5-class flower classification, compiles the model, trains it with callbacks (Early Stopping, ReduceLROnPlateau), evaluates performance, plots training curves, and finally saves the trained model as `model.h5`.
2.  **Model Loading (Flask App):** The `app.py` script loads the pre-trained Keras model (`model.h5`) into memory when the Flask application starts. Logging is configured to monitor the loading process and any errors.
3.  **Web Interface (HTML + JavaScript):**
    *   `index.html` provides the user interface with a drag-and-drop area or file input for image uploads, an image preview area, a loading spinner, and a div to display results.
    *   JavaScript embedded in `index.html` handles frontend interactions:
        *   Implementing drag-and-drop functionality for image files.
        *   Handling file selection via the hidden input.
        *   Displaying a preview of the selected image.
        *   Showing a loading spinner while the image is being processed.
        *   Sending the image file to the Flask backend using `axios` (an AJAX library).
        *   Receiving the JSON response containing class predictions and confidences.
        *   Updating the `results` div to display the top predictions using visually appealing cards and confidence bars.
        *   Showing a "Upload New Image" button after an image is processed to clear the interface and allow a new upload.
        *   Handling basic frontend error display.
4.  **Prediction Endpoint (`/` with POST):**
    *   The Flask `upload_file` function handles both GET requests (serving `index.html`) and POST requests (receiving the uploaded image).
    *   For POST requests, it checks if a file is included and if the file type is allowed.
    *   The uploaded file is securely saved to the `static/uploads` directory.
    *   The `prepare_image` function reads the saved image using OpenCV, converts it to RGB, applies minor random transformations (rotation, brightness - designed to handle slight variations in user uploads, although the core augmentation happened in notebook training), resizes it to the model's expected input size (224x224), applies EfficientNet-specific preprocessing, and adds a batch dimension.
    *   The loaded Keras model's `predict` method is called with the prepared image data.
    *   `process_predictions` (though minimal in this version as the model has softmax) might handle temperature scaling or other post-processing if needed (currently adds slight noise and renormalizes for robust probability display).
    *   The top 3 predictions (class name and confidence) are extracted from the model's output.
    *   A JSON response is returned to the frontend containing the uploaded filename and the sorted top predictions.
    *   Robust error handling is included at multiple stages.

## Model Performance (from Notebook)

The notebook `Flowers-Classification-Using-CNN copy.ipynb` includes detailed evaluation of the trained model. The performance metrics after training on the dataset are reported:

*   **Training, Validation, and Test Accuracy:** The notebook shows a high accuracy on the test set (~0.86 or 86% in the filename, actual value from notebook output is needed).
*   **Loss Curves:** Plots show the training and validation loss decreasing over epochs, indicating that the model learned effectively and Early Stopping likely prevented significant overfitting.
*   **Accuracy Curves:** Plots show training and validation accuracy increasing, converging towards a good performance level.
*   **Precision, Recall, AUC Curves:** These plots provide a more nuanced view of the model's performance for each class, especially useful if there was class imbalance (though augmentation and EfficientNet help).

These metrics suggest the model has learned to classify the five flower species with good accuracy and generalization capabilities on this specific dataset.

## Web Application Usage

1.  **Start the Flask server:** Follow the installation steps and run `python app.py` in your terminal. Ensure your virtual environment is activated and you are in the project's root directory.
2.  **Open in browser:** Navigate to `http://127.0.0.1:5000/` (or the address specified by Flask) in your web browser.
3.  **Upload Image:** Drag and drop a flower image onto the designated area, or click the area to select a file using your file browser.
4.  **Processing:** The application will show a preview of your image and a loading spinner while the image is sent to the server, processed, and the model makes a prediction.
5.  **View Results:** Once the analysis is complete, the loading spinner will disappear, and the top 3 predicted flower species along with their confidence percentages will be displayed in styled cards.
6.  **Upload New Image:** Click the "Upload New Image" button that appears after a result is shown to clear the interface and upload another image.
