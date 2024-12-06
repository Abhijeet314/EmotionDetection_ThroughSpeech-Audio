# Emotion Detection Through Speech/Audio

This project is an application for detecting human emotions such as **Happy**, **Sad**, **Angry**, and more by analyzing speech/audio files. The application is built using **Python**, **TensorFlow**, **Keras**, and **Streamlit**. It employs an **LSTM (Long Short-Term Memory)** model for sequential data processing and uses **Librosa** for audio feature extraction.

---

## Features
- Upload `.wav` or `.mp3` audio files.
- Extract and preprocess audio features using **MFCC (Mel-frequency cepstral coefficients)**.
- Predict emotions using a trained LSTM model.
- Visualize the audio data using waveforms and spectrograms.
- User-friendly web interface built with **Streamlit**.

---

## How It Works

1. **Audio File Upload**
   - Users upload an audio file through the web interface.
   - The file is saved to the `uploads` directory for processing.

2. **Feature Extraction**
   - Using **Librosa**, the audio file is loaded, and MFCC features are extracted. MFCC is a compact representation of the power spectrum and is widely used in speech and audio processing.
   - Features are normalized and reshaped to fit the LSTM model's input requirements.

3. **Prediction**
   - The preprocessed features are passed to the trained LSTM model (`sp.h5`) for emotion classification.
   - The model outputs probabilities for each emotion, and the emotion with the highest probability is displayed.

4. **Visualization**
   - Visualizations include:
     - **Waveplots** to display amplitude variations over time.
     - **Spectrograms** to analyze frequency and energy distribution of the audio file.

5. **Real-time Interaction**
   - Streamlit provides an intuitive interface for users to upload audio files and view predictions interactively.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Steps

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the trained model file (`sp.h5`) in the project directory.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Access the application in your browser (default: `http://localhost:8501`).
2. Upload a `.wav` or `.mp3` file using the file uploader.
3. Click the **Classify** button to predict the emotion.
4. View the predicted emotion and audio visualizations.

---

## File Structure
```
project_directory/
|-- app.py              # Main Streamlit application file
|-- sp.h5               # Pre-trained LSTM model file
|-- requirements.txt    # List of Python dependencies
|-- uploads/            # Directory for uploaded audio files
|-- README.md           # Project documentation
```

---

## Dependencies

- **TensorFlow**: For building and loading the LSTM model.
- **Keras**: High-level API for neural network modeling.
- **Streamlit**: For building the web application.
- **Librosa**: For audio feature extraction.
- **NumPy**: For numerical computations.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Add support for more emotions.
- Improve visualization with additional audio analysis graphs.
- Enhance the model's accuracy with a larger and more diverse dataset.
- Deploy the application to a cloud platform for wider accessibility.

---

## Acknowledgments
- **Librosa** documentation for audio feature extraction techniques.
- Tutorials on LSTM for sequential data processing.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Contributing
Contributions are welcome! If you'd like to add features or fix bugs, feel free to fork the repository and submit a pull request.

