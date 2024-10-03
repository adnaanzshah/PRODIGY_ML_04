
## 📖 Overview
The **Hand Gesture Recognition System** aims to leverage deep learning techniques to identify and classify various hand gestures from image or video data. This model enables intuitive human-computer interaction and serves as a key component for gesture-based control systems. The implementation focuses on utilizing a Convolutional Neural Network (CNN) to accurately recognize and interpret hand gestures in real-time applications.

## 📊 Dataset
The dataset comprises near-infrared images captured by the **Leap Motion** sensor. It includes **10 unique hand gestures** performed by **10 different subjects**, ensuring diversity in gesture styles and orientations. Each gesture is organized into folders corresponding to the subjects (00 to 09) and specific gestures (e.g., "01_palm," "02_l," "10_down"). This structured organization facilitates comprehensive training and evaluation of the model.

- **Dataset Link**: [Leap Gesture Recognition Dataset](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)

## 🛠️ Technologies Used
1. **TensorFlow** and **Keras**: Open-source ML frameworks for efficient model development.
2. **Convolutional Neural Network (CNN)**: Architecture employed for robust image-based hand gesture recognition.
3. **ImageDataGenerator**: Keras utility for efficient image loading and preprocessing.
4. **Model Training**: Utilizes the **Adam** optimizer and **categorical cross-entropy** loss for training the CNN model.
5. **Class Label Encoding**: Scikit-learn's **LabelEncoder** for encoding class labels.
6. **Python**: Primary programming language for data processing and model development.
7. **NumPy**: Library for numerical computations and array operations.
8. **Matplotlib**: Library for data visualization.

## ⚙️ Workflow Steps
1. **Data Preparation**:
   - Collect or use the Leap Gesture Recognition dataset containing diverse hand gesture images.
   - Organize the dataset into subdirectories, each corresponding to a specific hand gesture class.
  
2. **Model Architecture**:
   - Design a CNN architecture suitable for image classification tasks.
   - Select appropriate hyperparameters, including image size, batch size, and the number of epochs for training.

3. **Data Preprocessing**:
   - Utilize **ImageDataGenerator** for loading and preprocessing images.
   - Apply data augmentation techniques to enhance model generalization and robustness.

4. **Model Training**:
   - Train the CNN model on the preprocessed dataset.
   - Monitor validation performance to adjust hyperparameters and improve model accuracy.

5. **Prediction**:
   - Develop a function to make predictions on individual images using the trained model.
   - Validate the model’s accuracy using a separate test dataset.

6. **Model Saving and Loading**:
   - Save the trained model for future use.
   - Implement functionality to load the model for making predictions on new data.

## 🏁 Conclusion
The developed **Hand Gesture Recognition System** provides a solid foundation for intuitive human-computer interaction and gesture-based control systems. By leveraging deep learning, particularly CNNs, the model effectively learns intricate patterns in hand gestures, resulting in a robust and accurate solution for real-world applications.

## 🔗 Repository Link
Access the project repository here: [adnaanzshah/PRODIGY_ML_04](https://github.com/adnaanzshah/PRODIGY_ML_04)
