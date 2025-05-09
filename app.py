import os
import tempfile
import base64
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from faster_whisper import WhisperModel
import jiwer
from pydub import AudioSegment
import numpy as np
from werkzeug.utils import secure_filename
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import torch
from multiprocessing import Pool, cpu_count
from joblib import Memory
import threading
from flask_sock import Sock
import noisereduce as nr
import soundfile as sf
import re
import json
from pathlib import Path
import webrtcvad
import array
import struct
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

# Configure model caching
memory = Memory(location='./cache', verbose=0)

# Initialize VAD
vad = webrtcvad.Vad(3)  # Aggressive filtering

def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int) -> bytes:
    """Generate audio frames from PCM audio data."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n <= len(audio):
        yield audio[offset:offset + n]
        timestamp += duration
        offset += n

def vad_collector(sample_rate: int, frame_duration_ms: int, padding_duration_ms: int, 
                 frames: List[bytes]) -> List[bytes]:
    """Filter out non-speech frames using VAD."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    
    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        
        if not triggered:
            ring_buffer.append((frame, timestamp, is_speech))
            num_voiced = len([f for f, _, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, _, _ in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
                continue
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, timestamp, is_speech))
            num_unvoiced = len([f for f, _, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield voiced_frames
                ring_buffer.clear()
                voiced_frames = []
    
    if voiced_frames:
        yield voiced_frames

def process_audio_chunk(chunk_data: Tuple[str, int, int, int]) -> str:
    """Process a single audio chunk with VAD and transcription."""
    audio_path, start_ms, end_ms, overlap_ms = chunk_data
    try:
        # Load and process audio chunk
        audio = AudioSegment.from_file(audio_path)
        chunk = audio[start_ms:end_ms + overlap_ms]
        
        # Convert to raw PCM for VAD
        raw_data = chunk.raw_data
        sample_rate = chunk.frame_rate
        
        # Apply VAD
        frames = list(frame_generator(30, raw_data, sample_rate))
        voiced_segments = list(vad_collector(sample_rate, 30, 300, frames))
        
        if not voiced_segments:
            return ""
        
        # Save processed chunk
        temp_chunk_path = os.path.join(tempfile.gettempdir(), f"chunk_{os.urandom(4).hex()}.wav")
        processed_audio = AudioSegment.from_raw(
            b''.join(voiced_segments),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1
        )
        processed_audio.export(temp_chunk_path, format='wav')
        
        # Transcribe
        segments, _ = model.transcribe(temp_chunk_path)
        result_text = ' '.join([segment.text for segment in segments])
        
        # Cleanup
        try:
            os.unlink(temp_chunk_path)
        except Exception:
            pass
            
        return result_text
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return ""

def process_long_audio(audio_path: str, chunk_length_sec: int = 30, overlap_sec: int = 1) -> str:
    """Process long audio file in chunks with parallel processing and VAD."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_ms = len(audio)
        chunk_length_ms = chunk_length_sec * 1000
        overlap_ms = overlap_sec * 1000
        
        # Prepare chunks for parallel processing
        chunks = []
        for start_ms in range(0, duration_ms, chunk_length_ms):
            end_ms = min(start_ms + chunk_length_ms, duration_ms)
            chunks.append((audio_path, start_ms, end_ms, overlap_ms))
        
        # Process chunks in parallel
        with Pool(cpu_count()) as pool:
            transcriptions = pool.map(process_audio_chunk, chunks)
            
            # Update progress
            total_chunks = len(chunks)
            for i, _ in enumerate(transcriptions):
                progress = int((i + 1) / total_chunks * 100)
                broadcast_progress(progress)
        
        return ' '.join(filter(None, transcriptions))
    except Exception as e:
        logger.error(f"Error in long audio processing: {str(e)}")
        raise

@memory.cache
def load_cached_model():
    """Load and cache the Whisper model with CPU optimizations"""
    model_size = "base"  # Using base model for better accuracy/speed balance
    logger.info('Using CPU-optimized Whisper model')
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_count(),
        num_workers=cpu_count()
    )
    return model

# Initialize model
model = load_cached_model()

# Configure upload folder
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def normalize_text(text):
    """Apply text normalization rules"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize numbers
        text = re.sub(r'(\d+)(?:st|nd|rd|th)', r'\1', text)
        
        # Normalize common abbreviations
        text = re.sub(r'\b(?:mr|mrs|dr|prof|inc|ltd|co|corp)\.', r'\1', text)
        
        # Remove punctuation except for sentence boundaries
        text = re.sub(r'[^\w\s\.]', '', text)
        
        return text
    except Exception as e:
        logger.error(f"Error in text normalization: {str(e)}")
        return text

def denoise_audio(audio_path):
    """Apply noise reduction using noisereduce"""
    try:
        # Read audio file
        data, sample_rate = sf.read(audio_path)
        
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=data,
            sr=sample_rate,
            prop_decrease=0.75,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )
        
        # Save denoised audio
        denoised_path = audio_path.replace('.wav', '_denoised.wav')
        sf.write(denoised_path, reduced_noise, sample_rate)
        
        return denoised_path
    except Exception as e:
        logger.error(f"Error in denoising: {str(e)}")
        return audio_path

def contextual_correction(text):
    """Apply BERT-based contextual correction"""
    try:
        # Tokenize and find potential errors
        tokens = bert_tokenizer.tokenize(text)
        corrected_tokens = []
        
        for i, token in enumerate(tokens):
            # Mask token and get prediction
            masked_tokens = tokens.copy()
            masked_tokens[i] = bert_tokenizer.mask_token
            inputs = bert_tokenizer.encode(' '.join(masked_tokens), return_tensors='pt')
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = bert_model(inputs)
                predictions = outputs[0]
            
            # Get top prediction
            masked_index = (inputs == bert_tokenizer.mask_token_id).nonzero().item()
            predicted_token = bert_tokenizer.decode([predictions[0, masked_index].argmax().item()])
            
            # Use prediction if confidence is high
            if predictions[0, masked_index].max().item() > 0.8:
                corrected_tokens.append(predicted_token)
            else:
                corrected_tokens.append(token)
        
        return ' '.join(corrected_tokens)
    except Exception as e:
        logger.error(f"Error in contextual correction: {str(e)}")
        return text

def normalize_text(text):
    """Apply text normalization"""
    try:
        return normalize(text)
    except Exception as e:
        logger.error(f"Error in text normalization: {str(e)}")
        return text

def log_transcription_errors(original_text, wer_text, reference_text=None):
    """Log detailed error analysis using WER-normalized text."""
    try:
        norm_ref = normalize_for_wer(reference_text) if reference_text else None
        error_log = {
            'timestamp': time.time(),
            'original_text': original_text,
            'wer_text': wer_text,
            'reference_text': reference_text,
            'wer_reference': norm_ref,
            'errors': {}
        }
        if reference_text:
            transformation = lambda x: x.split()
            measures = jiwer.compute_measures(norm_ref, wer_text, truth_transform=transformation, hypothesis_transform=transformation)
            error_log['errors'] = {
                'wer': measures['wer'],
                'substitutions': measures['substitutions'],
                'deletions': measures['deletions'],
                'insertions': measures['insertions']
            }
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'transcription_errors.json'
        with open(log_file, 'a') as f:
            f.write(json.dumps(error_log) + '\n')
    except Exception as e:
        logger.error(f"Error in logging transcription errors: {str(e)}")

def process_audio_file(audio_file):
    """Process audio file with optimized pipeline"""
    try:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"audio_{int(time.time())}_{os.urandom(4).hex()}.wav")
        audio_file.save(temp_file_path)
        
        # Convert to WAV if needed
        if not temp_file_path.endswith('.wav'):
            audio = AudioSegment.from_file(temp_file_path)
            wav_path = temp_file_path + '.wav'
            audio.export(wav_path, format='wav')
            os.unlink(temp_file_path)
            temp_file_path = wav_path
        
        # Preprocess audio
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Normalize to 16kHz mono
        audio.export(temp_file_path, format='wav')
        
        # Apply noise reduction
        denoised_path = denoise_audio(temp_file_path)
        
        # Process based on duration
        if len(audio) > 120 * 1000:  # 120 seconds
            result_text = process_long_audio(denoised_path)
        else:
            segments, _ = model.transcribe(denoised_path)
            result_text = ' '.join([segment.text for segment in segments])
        
        # Post-process for display
        final_text = format_transcription(result_text)
        
        # WER normalization and logging
        wer_text = normalize_for_wer(result_text)
        log_transcription_errors(result_text, wer_text)
        
        # Clean up
        try:
            os.unlink(temp_file_path)
            if denoised_path != temp_file_path:
                os.unlink(denoised_path)
        except Exception as e:
            logger.error(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
            
        logger.info(f"Final transcription output: {repr(final_text)}")
        return final_text
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def process_base64_audio(base64_audio):
    """Process base64 encoded audio data and return transcription"""
    try:
        audio_data = base64.b64decode(base64_audio)
        temp_dir = tempfile.gettempdir()
        webm_path = os.path.join(temp_dir, f"recording_{int(time.time())}_{os.urandom(4).hex()}.webm")
        wav_path = webm_path.replace('.webm', '.wav')
        
        # Save and convert to WAV
        with open(webm_path, 'wb') as f:
            f.write(audio_data)
        audio = AudioSegment.from_file(webm_path, format='webm')
        audio = audio.set_frame_rate(16000).set_channels(1)  # Normalize to 16kHz mono
        audio.export(wav_path, format='wav')
        
        # Clean up webm file
        try:
            os.unlink(webm_path)
        except Exception as e:
            print(f"Warning: Could not delete webm file {webm_path}: {e}")
        
        # Check if audio is long
        if len(audio) > 120 * 1000:  # 120 seconds
            result_text = process_long_audio(wav_path)
        else:
            segments, _ = model.transcribe(wav_path)
            result_text = ' '.join([segment.text for segment in segments])
        
        # Clean up wav file
        try:
            os.unlink(wav_path)
        except Exception as e:
            print(f"Warning: Could not delete wav file {wav_path}: {e}")
            
        return result_text
    except Exception as e:
        print(f"Error processing base64 audio: {str(e)}")
        raise

def calculate_wer_stats(reference, hypothesis):
    """Calculate WER and get detailed statistics"""
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.SentencesToListOfWords(),
        jiwer.RemovePunctuation()
    ])
    
    reference_words = transformation(reference)
    hypothesis_words = transformation(hypothesis)
    
    wer = jiwer.wer(reference, hypothesis, truth_transform=transformation, hypothesis_transform=transformation)
    
    # Get detailed statistics
    measures = jiwer.compute_measures(reference, hypothesis, truth_transform=transformation, hypothesis_transform=transformation)
    
    # Find substitutions
    substitutions = []
    ref_idx = hyp_idx = 0
    while ref_idx < len(reference_words) and hyp_idx < len(hypothesis_words):
        if reference_words[ref_idx] != hypothesis_words[hyp_idx]:
            substitutions.append({
                'expected': reference_words[ref_idx],
                'transcribed': hypothesis_words[hyp_idx]
            })
        ref_idx += 1
        hyp_idx += 1
    
    return {
        'wer': wer * 100,  # Convert to percentage
        'substitutions': measures['substitutions'],
        'deletions': measures['deletions'],
        'insertions': measures['insertions'],
        'substitution_details': substitutions
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe_upload', methods=['POST'])
def transcribe_upload():
    try:
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files['audio']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        logger.debug(f"Saving file to: {temp_path}")
        file.save(temp_path)
        try:
            if not os.path.exists(temp_path):
                logger.error(f"File not found after saving: {temp_path}")
                return jsonify({'error': 'File could not be saved'}), 500
            # Transcribe the audio
            logger.debug("Starting transcription")
            segments, _ = model.transcribe(temp_path)
            result_text = ' '.join([segment.text for segment in segments])
            final_text = format_transcription(result_text)
            logger.info(f"Returned to frontend: {repr(final_text)}")
            logger.debug("Transcription completed successfully")
            return jsonify({'text': final_text})
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe_recorded', methods=['POST'])
def transcribe_recorded():
    try:
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files['audio']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        logger.debug(f"Saving file to: {temp_path}")
        file.save(temp_path)
        try:
            if not os.path.exists(temp_path):
                logger.error(f"File not found after saving: {temp_path}")
                return jsonify({'error': 'File could not be saved'}), 500
            # Transcribe the audio
            logger.debug("Starting transcription")
            segments, _ = model.transcribe(temp_path)
            result_text = ' '.join([segment.text for segment in segments])
            final_text = format_transcription(result_text)
            logger.info(f"Returned to frontend: {repr(final_text)}")
            logger.debug("Transcription completed successfully")
            return jsonify({'text': final_text})
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.debug(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        if not data or 'reference' not in data or 'hypothesis' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        reference = data['reference']
        hypothesis = data['hypothesis']
        
        # Calculate WER and get statistics
        stats = calculate_wer_stats(reference, hypothesis)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/wer_analysis', methods=['POST'])
def wer_analysis():
    try:
        data = request.get_json()
        reference = data.get('reference', '')
        hypothesis = data.get('hypothesis', '')
        model_name = data.get('model_name', None)
        file_name = data.get('file_name', None)
        identifier = extract_identifier(model_name, file_name)

        # Calculate WER and get statistics
        stats = calculate_wer_stats(reference, hypothesis)

        # Generate all required plots
        radar_chart = plot_radar(stats, identifier)
        prf_bar = plot_prf_bar(stats, identifier)
        confusion_matrix_img = plot_confusion_matrix(stats, identifier)

        return jsonify({
            'identifier': identifier,
            'metrics': stats,
            'radar_chart': radar_chart,
            'prf_bar': prf_bar,
            'confusion_matrix': confusion_matrix_img,
        })
    except Exception as e:
        logger.error(f"WER analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- Data Processing ---
def extract_identifier(model_name=None, file_name=None):
    """Return a meaningful label for the model/file."""
    if model_name:
        return model_name
    if file_name:
        return file_name
    return "Unknown"

# --- Metrics Computation ---
def compute_wer_metrics(reference, hypothesis):
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.SentencesToListOfWords(),
        jiwer.RemovePunctuation()
    ])
    ref_words = transformation(reference)
    hyp_words = transformation(hypothesis)
    measures = jiwer.compute_measures(reference, hypothesis, truth_transform=transformation, hypothesis_transform=transformation)
    wer = measures['wer']
    subs = measures['substitutions']
    ins = measures['insertions']
    dels = measures['deletions']
    hits = measures['hits']
    total_words = len(ref_words)
    correct = hits

    # ML metrics
    TP = hits
    FP = ins
    FN = dels
    TN = 0  # Not well-defined for text, but included for completeness

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'wer': wer,
        'substitutions': subs,
        'insertions': ins,
        'deletions': dels,
        'correct': correct,
        'total_words': total_words,
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --- Graph Rendering ---
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def plot_radar(metrics, identifier):
    categories = ['Correct', 'Substitutions', 'Deletions', 'Insertions']
    values = [metrics['correct'], metrics['substitutions'], metrics['deletions'], metrics['insertions']]
    N = len(categories)
    values += values[:1]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=identifier)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f'Error Profile (Radar) - {identifier}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return fig_to_base64(fig)

def plot_prf_bar(metrics, identifier):
    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.bar(['Precision', 'Recall', 'F1-score'], [metrics['precision'], metrics['recall'], metrics['f1']], color=['#2980b9', '#27ae60', '#e67e22'])
    ax.set_ylim(0, 1)
    ax.set_title(f'Precision, Recall, F1-score - {identifier}')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    return fig_to_base64(fig)

def plot_confusion_matrix(metrics, identifier):
    fig, ax = plt.subplots(figsize=(4,4))
    cm = np.array([[metrics['TP'], metrics['FP']], [metrics['FN'], metrics['TN']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Predicted Correct (TP)', 'Predicted Error (FP)'],
                yticklabels=['Actual Correct (TP)', 'Actual Error (FN)'])
    ax.set_title(f'Confusion Matrix (Word Level) - {identifier}')
    # Add mapping note
    ax.text(0, -0.5, "Mapping: Insertions=FP, Deletions=FN, Correct=TP", fontsize=8, color='gray')
    return fig_to_base64(fig)

# WebSocket for progress updates
progress_clients = set()
progress_lock = threading.Lock()

def broadcast_progress(progress):
    """Broadcast progress to all connected clients"""
    with progress_lock:
        for client in progress_clients:
            try:
                client.send(str(progress))
            except Exception as e:
                logger.error(f"Error sending progress: {e}")
                progress_clients.remove(client)

@sock.route('/progress')
def progress(ws):
    """WebSocket endpoint for progress updates"""
    progress_clients.add(ws)
    try:
        while True:
            # Keep connection alive
            ws.receive()
    except Exception:
        pass
    finally:
        progress_clients.remove(ws)

def format_transcription(text):
    # Split on sentence-ending punctuation (., ?, !)
    sentences = re.split(r'[.?!]+', text)
    formatted = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Remove all punctuation (including apostrophes)
        s = re.sub(r"[^\w\s]", '', s)
        # Lowercase, then capitalize first letter
        s = s.lower()
        s = s[0].upper() + s[1:] if s else ''
        formatted.append(s)
    return '\n'.join(formatted)

def normalize_for_wer(text):
    # Remove apostrophes and punctuation, lowercase
    text = re.sub(r"[^\w\s]", '', text).lower()
    return re.sub(r"\s+", " ", text).strip()

if __name__ == '__main__':
    app.run(debug=True) 