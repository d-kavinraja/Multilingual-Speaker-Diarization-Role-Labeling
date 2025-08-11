
import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from datetime import datetime
import warnings
import re
import tempfile
import streamlit as st

# --- Import Libraries (assumed to be installed via requirements.txt) ---
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from langdetect import detect, DetectorFactory

# Set seeds for reproducibility
warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)
DetectorFactory.seed = 0

# Your HealthPilotDiarizer class (copied from your script with minor adjustments)
# No major changes are needed inside the class itself.
class HealthPilotDiarizer:
    """
    A complete pipeline for multilingual speaker diarization and role labeling
    for medical conversations (e.g., Doctor-Patient).
    """
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.diarization_pipeline = None
        self.asr_pipeline = None
        self.setup_models()

        self.medical_terms = {
             'english': ['pain', 'symptom', 'diagnosis', 'treatment', 'medication', 'doctor', 'blood pressure', 'fever', 'headache', 'prescription', 'therapy', 'surgery', 'clinic', 'hospital', 'medicine', 'tablets', 'injection', 'examination', 'test', 'report', 'disease', 'condition', 'history'],
             'hindi': ['dard', 'bukhar', 'dawai', 'ilaaj', 'doctor', 'mareez', 'aspataal', 'clinic', 'goli', 'injection', 'jaanch', 'report', 'bimari', 'takleef'],
             'tamil': ['vali', 'kaachal', 'marundhu', 'sihichai', 'maruthuvar', 'noyali', 'maruthuvamanai', 'klinik', 'maathirai', 'oosi', 'parisodhanai', 'arikkai', 'noi', 'pirachanai']
        }
        self.doctor_patterns = [r'\b(how|what|when|where|which|tell me|describe|any|do you|is there)\b', r'\?(.*)', r'\b(i will|we need|you should|i recommend|let me check)\b']

    def setup_models(self):
        """Loads and prepares all required models."""
        st.write("🔄 Setting up models...")
        try:
            st.write("  - Loading Speaker Diarization model (pyannote/speaker-diarization-3.1)...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            self.diarization_pipeline.to(self.device)
            st.write("  ✅ Diarization model loaded.")
        except Exception as e:
            st.error(f"❌ Error loading diarization model: {e}")
            st.error("Please ensure you have accepted the user agreement on Hugging Face for this model and your token is correct.")
            st.stop()

        try:
            st.write("  - Loading ASR model (openai/whisper-base)...")
            self.asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device=self.device
            )
            st.write("  ✅ ASR model loaded.")
        except Exception as e:
            st.error(f"❌ Error loading ASR model: {e}")
            st.stop()
        st.write("--- Models setup complete ---")

    def process_audio(self, audio_path: str, encounter_id: str = "enc_default_001"):
        if not os.path.exists(audio_path):
            st.error(f"❌ File not found at {audio_path}")
            return None

        status_placeholder = st.empty()

        status_placeholder.info("[Step 1/4] Performing Speaker Diarization...")
        diarization_result = self.diarization_pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segments.append({'speaker': speaker, 'start': turn.start, 'end': turn.end})
        
        # Guard against no speakers found
        if not segments:
            status_placeholder.warning("Could not detect any speech segments in the audio.")
            return None

        status_placeholder.info(f"[Step 2/4] Transcribing {len(segments)} audio segments...")
        all_transcribed = self.transcribe_segments(audio_path, segments)

        # *** MODIFIED PART: Filter out failed or very short transcriptions ***
        transcribed_segments = [
            s for s in all_transcribed if s.get("text") and len(s["text"].strip()) > 2
        ]
        
        num_filtered = len(all_transcribed) - len(transcribed_segments)
        status_placeholder.info(f"Filtered out {num_filtered} noisy or errored segments.")

        status_placeholder.info("[Step 3/4] Classifying speaker roles...")
        speaker_roles = self.classify_roles(transcribed_segments)
        
        status_placeholder.info("[Step 4/4] Formatting output...")
        # *** MODIFIED PART: Clean up the detected languages list ***
        detected_languages = list(set(
            s.get('lang') for s in transcribed_segments if s.get('lang') and s.get('lang') not in ['err', 'n/a']
        ))
        result = self.format_output(encounter_id, transcribed_segments, speaker_roles, detected_languages)
        
        status_placeholder.success("🏁 Processing Complete!")
        return result

    def transcribe_segments(self, audio_path: str, segments: list) -> list:
        try:
            waveform, original_sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            st.error(f"❌ Could not load audio file: {e}")
            return segments

        target_sample_rate = 16000
        if original_sample_rate != target_sample_rate:
            resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
        
        sample_rate = target_sample_rate
        progress_bar = st.progress(0, text=f"Transcribing 0/{len(segments)} segments...")

        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = waveform[0, start_sample:end_sample].numpy()
            
            segment_duration = len(segment_audio) / sample_rate
            if segment_duration < 0.2:
                segment['text'] = ""
                segment['lang'] = "n/a"
                continue

            try:
                generate_kwargs = {"task": "transcribe"}
                output = self.asr_pipeline(
                    segment_audio.copy(),
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs
                )
                transcription = output['text'].strip()
                segment['text'] = transcription
                
                if len(transcription) > 10:
                    try:
                        segment['lang'] = detect(transcription)
                    except:
                        segment['lang'] = 'en'
                else:
                    segment['lang'] = 'en'

            except Exception:
                # *** MODIFIED PART: Set text to None on error to filter later ***
                segment['text'] = None
                segment['lang'] = "err"
            
            progress_bar.progress((i + 1) / len(segments), text=f"Transcribing {i+1}/{len(segments)} segments...")
        
        progress_bar.empty()
        return segments

    def classify_roles(self, segments: list) -> dict:
        speaker_stats = {}
        unique_speakers = sorted(list(set(s['speaker'] for s in segments)))
        for speaker in unique_speakers:
            speaker_segments = [s for s in segments if s['speaker'] == speaker]
            # Ensure text exists before joining
            full_text = " ".join([s.get('text', '') for s in speaker_segments if s.get('text')]).lower()
            if not full_text: continue
            all_medical_terms = self.medical_terms['english'] + self.medical_terms['hindi'] + self.medical_terms['tamil']
            medical_term_count = sum([full_text.count(term) for term in all_medical_terms])
            question_pattern_count = sum([len(re.findall(p, full_text)) for p in self.doctor_patterns])
            word_count = len(full_text.split())
            sentence_count = len(re.split(r'[.!?]+', full_text))
            medical_score = (medical_term_count / word_count * 10) if word_count > 0 else 0
            question_score = (question_pattern_count / sentence_count) if sentence_count > 0 else 0
            clinician_score = (medical_score * 0.6) + (question_score * 0.4)
            speaker_stats[speaker] = {'score': clinician_score}
        if not speaker_stats: return {sp: {'role': 'other', 'confidence': 0.5} for sp in unique_speakers}
        scores = [(sp, data['score']) for sp, data in speaker_stats.items()]
        if len(scores) >= 2: # More robust check for 2 or more speakers
            scores.sort(key=lambda x: x[1], reverse=True)
            clinician_sp, clinician_score = scores[0]
            patient_sp, _ = scores[1]
            return {
                clinician_sp: {'role': 'clinician', 'confidence': round(min(0.5 + clinician_score, 0.98), 2)},
                patient_sp: {'role': 'patient', 'confidence': round(max(0.98 - clinician_score, 0.75), 2)}
            }
        elif len(scores) == 1: # Handle case with only one speaker
             sp, score = scores[0]
             role = 'clinician' if score > 0.3 else 'patient'
             return {sp: {'role': role, 'confidence': 0.65}}
        else:
            return {} # No valid speakers with text
            
    def format_output(self, encounter_id: str, segments: list, speaker_roles: dict, languages: list) -> dict:
        speakers_list = [{"id": sp_id, "role": role_info.get("role", "other"), "confidence": role_info.get("confidence", 0.5)} for sp_id, role_info in speaker_roles.items()]
        # The segments list is already filtered, so this is fine
        segments_list = [{"id": f"seg_{i+1:04d}", "speaker": s["speaker"], "startSec": round(s["start"], 2), "endSec": round(s["end"], 2), "text": s.get("text", ""), "lang": s.get("lang", "n/a")} for i, s in enumerate(segments)]
        return {
            "encounterId": encounter_id, "detectedLanguages": sorted(languages) if languages else ["en"],
            "speakers": speakers_list, "segments": segments_list,
            "createdAt": datetime.now().isoformat() + "Z"
        }

# --- Streamlit UI ---

st.set_page_config(page_title="HealthPilot Diarization", layout="wide")
st.title("🏥 HealthPilot: Medical Conversation Diarization")
st.markdown("Upload a multilingual audio recording of a medical consultation (e.g., Doctor-Patient) to transcribe and identify who spoke when.")

# --- Helper function to load models with caching ---
@st.cache_resource
def load_diarizer(token):
    """Loads the HealthPilotDiarizer instance and caches it."""
    try:
        diarizer = HealthPilotDiarizer(hf_token=token)
        return diarizer
    except Exception as e:
        st.error(f"Failed to initialize models: {e}")
        return None

# --- Main App Logic ---

# 1. Get Hugging Face Token
st.sidebar.header("Configuration")
try:
    # Try to get token from st.secrets
    hf_token = st.secrets["HF_TOKEN"]
    st.sidebar.success("✅ Hugging Face token loaded from secrets.")
except (FileNotFoundError, KeyError):
    st.sidebar.warning("HF_TOKEN not found in secrets. Please enter it below.")
    hf_token = st.sidebar.text_input("Enter your Hugging Face Token:", type="password")

if not hf_token:
    st.warning("Please provide a Hugging Face token in the sidebar to proceed.")
    st.stop()

# 2. Load Models (will be cached)
diarizer = load_diarizer(hf_token)
if not diarizer:
    st.stop()

# 3. File Uploader
uploaded_file = st.file_uploader(
    "Choose an audio file...",
    type=['mp3', 'wav', 'm4a', 'flac']
)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Process Audio File 🎙️"):
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            temp_audio_path = tmpfile.name

        try:
            with st.spinner('Processing audio... This may take a few minutes.'):
                # Process the audio file
                encounter_id = f"enc_streamlit_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                result = diarizer.process_audio(temp_audio_path, encounter_id=encounter_id)
            
            if result:
                st.balloons()
                st.header("📊 Results Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🗣️ Speaker Roles")
                    if result['speakers']:
                        for speaker in result['speakers']:
                            st.write(f"- **{speaker['id']}**: Identified as **{speaker['role']}** (Confidence: {speaker['confidence']})")
                    else:
                        st.write("No speakers identified after filtering.")

                with col2:
                    st.subheader("🌐 Detected Languages")
                    st.write(f"Languages found: **{', '.join(result['detectedLanguages'])}**")
                
                st.divider()

                # Display full transcription
                st.header("📝 Full Transcription")
                if result['segments']:
                    for segment in result['segments']:
                        role = next((sp['role'] for sp in result['speakers'] if sp['id'] == segment['speaker']), 'Unknown')
                        st.markdown(f"**[{segment['startSec']:.2f}s - {segment['endSec']:.2f}s] {role.capitalize()} ({segment['speaker']}):** {segment['text']}")
                else:
                    st.warning("No valid speech segments could be transcribed from the audio.")
                
                # Provide a download link for the JSON
                st.download_button(
                    label="Download Full JSON Result",
                    data=json.dumps(result, indent=4, ensure_ascii=False),
                    file_name=f"{encounter_id}_diarization_result.json",
                    mime="application/json"
                )

                # Optional: Show the raw JSON in an expander
                with st.expander("Show Raw JSON Output"):
                    st.json(result)
            else:
                 st.error("Processing did not yield a result. The audio might be silent or too noisy.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        finally:
            # Clean up the temporary file
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
