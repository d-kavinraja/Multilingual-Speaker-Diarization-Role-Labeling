# HealthPilot: Multilingual Medical Conversation Diarization

An intelligent Streamlit application to transcribe and analyze multi-speaker medical consultations. This tool automatically identifies who spoke when (diarization), transcribes their speech (ASR), and assigns their role (Clinician or Patient), even in conversations that mix English and other languages like Hindi or Tamil.


---

## 🎯 Project Goal & Problem Statement

In modern healthcare, clinicians spend a significant amount of time on administrative tasks like note-taking, detracting from patient care. Ambient scribing solutions aim to solve this by automatically documenting conversations.

This project tackles a core component of ambient scribing: **knowing who spoke, when they spoke, and their role in the conversation.**

**Input:** A 3-5 minute audio file (`.wav`, `.mp3`) of a doctor-patient consultation.
**Output:** A structured JSON file containing:
*   Speaker-segmented transcription with precise timestamps.
*   Automatic role labels for each speaker (`clinician`, `patient`).
*   Language identification for each spoken segment.

This enables reliable record-keeping for medical, legal, and billing purposes, ultimately allowing doctors to focus more on their patients.

---

## ✨ Key Features

*   **🗣️ Speaker Diarization:** Uses `pyannote.audio` to determine the exact start and end times for each speaker's turn.
*   **🌐 Multilingual Transcription:** Employs OpenAI's `Whisper` model to accurately transcribe speech, tolerating code-switching (e.g., mixing English and Hindi).
*   **👨‍⚕️ Automatic Role Labeling:** A heuristic-based engine analyzes speech patterns (use of medical terms, question-asking) to intelligently label speakers as "clinician" or "patient".
*   **🚀 Interactive Web UI:** Built with Streamlit for a user-friendly interface to upload audio, view results, and download the output.
*   **🖥️ Local Model Processing:** All computation is done locally on your machine (CPU or GPU), ensuring data privacy and no reliance on paid APIs.
*   **📄 JSON Export:** Generates a clean, structured JSON output that can be easily integrated into other systems (like an Electronic Health Record - EHR).

---

## 🛠️ Technology Stack

*   **Backend & Machine Learning:**
    *   **PyTorch:** Core framework for deep learning models.
    *   **Hugging Face `transformers`:** For accessing the Whisper ASR model.
    *   **`pyannote.audio`:** State-of-the-art library for speaker diarization.
    *   **`torchaudio`:** For efficient audio loading and resampling.
    *   **`langdetect`:** For per-segment language identification.
*   **Frontend:**
    *   **Streamlit:** For creating the interactive web application.
*   **Core Models:**
    *   **Diarization:** `pyannote/speaker-diarization-3.1`
    *   **ASR:** `openai/whisper-base`
*   **Utilities:** NumPy, Re (Regular Expressions)

---

## ⚙️ System Architecture

The application follows a sequential pipeline to process the audio file:

1.  **Audio Upload (UI):** The user uploads an audio file via the Streamlit web interface.
2.  **Audio Pre-processing:** The file is loaded and resampled to **16kHz**, the standard rate required by the Whisper model for optimal performance.
3.  **Step 1: Speaker Diarization:** The `pyannote.audio` pipeline processes the entire audio to produce speaker turn segments (e.g., `SPEAKER_00` spoke from 2.5s to 5.8s).
4.  **Step 2: ASR Transcription:** Each individual speech segment is passed to the `Whisper` model, which transcribes the audio to text.
5.  **Step 3: Role Classification:** The transcribed text for each speaker is aggregated. A rule-based engine scores the text based on the frequency of medical terms and question patterns to assign the most likely role (`clinician` or `patient`).
6.  **Step 4: Output Formatting:** The results are compiled into a structured JSON file.
7.  **Results Display:** The final transcription, speaker roles, and detected languages are displayed on the Streamlit UI, with an option to download the complete JSON file.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

*   Python 3.9+
*   `ffmpeg`: Required by `torchaudio` to load various audio formats.
    *   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` folder to your system's PATH.
    *   **macOS (Homebrew):** `brew install ffmpeg`
    *   **Linux (apt):** `sudo apt update && sudo apt install ffmpeg`

### 2. Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a Python virtual environment:**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Create a `requirements.txt` file** with the following content:
    ```
    streamlit
    torch
    torchaudio
    pyannote.audio
    transformers
    langdetect
    accelerate
    numpy
    ```

4.  **Install the dependencies:**
    ```
    pip install -r requirements.txt
    ```

### 3. Hugging Face Setup

This project requires a Hugging Face access token to download the diarization model.

1.  **Get your Token:** Create a free account on [Hugging Face](https://huggingface.co/) and generate an access token at `Settings > Access Tokens`.
2.  **Accept User Agreement:** You **must** accept the user agreement for the following two models to be able to download them:
    *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (dependency)

### 4. Running the Application

1.  **Set your Hugging Face Token:** For local development, the best way is to use Streamlit's secrets management.
    *   Create a folder: `.streamlit`
    *   Inside it, create a file: `secrets.toml`
    *   Add your token to the file like this:
        ```
        HF_TOKEN = "hf_YourActualTokenGoesHere"
        ```

2.  **Launch the app:** Run the following command in your terminal.
    ```
    streamlit run app.py
    ```
    Your browser should automatically open with the application running.

---

## 📖 How to Use

1.  Launch the application using the command above.
2.  Use the file uploader to select an audio file (`.mp3`, `.wav`, etc.).
3.  The app will display an audio player to preview your file.
4.  Click the **"Process Audio File 🎙️"** button.
5.  Wait for the processing to complete. You will see status updates for each step.
6.  Once done, the results will be displayed, showing speaker roles, detected languages, and the full transcription.
7.  Click the **"Download Full JSON Result"** button to save the output file.

---

## 🔮 Future Improvements

*   **Advanced Role Classification:** Replace the heuristic-based role classifier with a fine-tuned ML model (e.g., a text classifier trained on medical conversation data) for higher accuracy.
*   **Dockerization:** Package the application into a Docker container for simplified deployment and environment consistency.
*   **Real-time Processing:** Implement real-time transcription and diarization using a streaming audio input.
*   **Database Integration:** Store results in a database (like PostgreSQL or MongoDB) to manage and query past consultations.

