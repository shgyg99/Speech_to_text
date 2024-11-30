# Speech-to-Text with Transformer Model🎙

This repository contains a Speech-to-Text model trained on the **LJ Speech Dataset**. The model utilizes:
- **CNN Feature Extractor** for extracting audio features.
- **Positional Encoding** to handle sequential data.
- **Transformer Architecture** for accurate transcription.

The project includes a web application deployed on **Hugging Face Spaces** and instructions for running the app using Docker.

---

## Live Demo

Click on the image below to try out the **Speech-to-Text Web App**:

[![Speech-to-Text App](https://github.com/shgyg99/Speech_to_text/blob/master/LiveDemo.png?raw=true)](https://shgyg99-audio-to-text.hf.space)

---

## Results

### Model Metrics
- **WER (Word Error Rate)**: `0.66` on test data.
- **Best Loss**: `0.128`
- **Worst Loss**: `1.851`

### Example Predictions
| **Generated**                                                                                       | **Target**                                                                                       | **Loss** |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|----------|
| *report of the president's commission on the assassination of president kennedy. bhe warren...*    | *report of the president's commission on the assassination of president kennedy. the warren...* | `0.128`  |
| *to clinngs sporting goods com  cr coocogov aniingwnh*                                             | *to klein's sporting goods co., of chicago, illinois.*                                          | `1.851`  |

---

## Run Locally with Docker

Follow the steps below to run the app using Docker:

1. **Clone this repository**:
    ```bash
    git clone https://github.com/shgyg99/Speech_to_text.git
    cd Speech_to_text
    ```

2. **Build the Docker image**:
    ```bash
    docker build -t speech-to-text-app .
    ```

3. **Run the Docker container**:
    ```bash
    docker run -p 8501:8501 speech-to-text-app
    ```

4. Open your browser and go to `http://localhost:8501` to access the app.

---

## Features

- **Interactive Interface**: Upload audio files and get transcriptions in real-time.
- **Transformer Model**: Leverages state-of-the-art techniques for transcription accuracy.
- **Docker Support**: Easily containerized for portability.

---

## Acknowledgments

- **Dataset**: [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- **Frameworks Used**: PyTorch, Streamlit, Docker

---

Feel free to fork, contribute, and create pull requests!
