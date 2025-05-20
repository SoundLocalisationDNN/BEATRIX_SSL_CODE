import time
import numpy as np
import sounddevice as sd
import requests  #For sending HTTP requests to the laptop server
from faster_whisper import WhisperModel
from yapper import PiperSpeaker, PiperVoiceUK
from scipy.signal import butter, lfilter

# parameters
SAMPLERATE = 16000
CHANNELS = 2  #Stereo audio
SILENCE_THRESHOLD = 0.0008
SILENCE_DURATION = 1
INITIAL_MAX_TIME = 8
MIN_SPEECH_DURATION = 1
LOWCUT = 300
HIGHCUT = 3400
LAPTOP_IP = "172.26.107.196"  # Replace with your laptop's IP
SERVER_PORT = 5000  # Flask server port

#Initialise models
ModelWhisper = WhisperModel("small.en", device="cpu", compute_type="int8")
lessac = PiperSpeaker(voice=PiperVoiceUK.ALBA)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, b, a):
    return lfilter(b, a, data, axis=0)

b, a = butter_bandpass(LOWCUT, HIGHCUT, SAMPLERATE, order=5)

def record_until_silence():
    print("Start speaking. Recording will end after silence.")
    recorded_chunks = []
    silence_start = None
    recording = True
    total_time = 0.0
    speech_time = 0.0
    chunk_duration = 0.25

    with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype='float32') as stream:
        while recording:
            chunk, overflow = stream.read(int(chunk_duration * SAMPLERATE))
            recorded_chunks.append(chunk)
            total_time += chunk_duration
            filtered_chunk = bandpass_filter(chunk, b, a)
            rms_channels = np.sqrt(np.mean(filtered_chunk ** 2, axis=0))
            avg_rms = np.mean(rms_channels)
            
            if avg_rms >= SILENCE_THRESHOLD:
                speech_time += chunk_duration

            if total_time >= INITIAL_MAX_TIME and speech_time <= MIN_SPEECH_DURATION:
                print("Not enough speech detected.")
                return np.concatenate(recorded_chunks, axis=0), False

            if avg_rms < SILENCE_THRESHOLD and speech_time >= MIN_SPEECH_DURATION:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Silent too long. Stopping.")
                    break
            else:
                silence_start = None

    return np.concatenate(recorded_chunks, axis=0), True

while True:
    audio, enough_speech = record_until_silence()
    print("Recording finished.")
    if not enough_speech:
        lessac.say("Goodbye.")
        break

    rms_overall = np.sqrt(np.mean(audio ** 2, axis=0))
    loudest_channel_index = np.argmax(rms_overall)
    audio_for_transcription = audio[:, loudest_channel_index]

    segments, info = ModelWhisper.transcribe(audio_for_transcription)
    transcription = " ".join(segment.text for segment in segments)
    print("Transcription:", transcription)
    
    if not transcription.strip() or "bye" in transcription.lower():
        lessac.say("Goodbye.")
        break

    try:
        response = requests.post(
            f"http://{LAPTOP_IP}:{SERVER_PORT}/send",
            json={"message": transcription},
            timeout=50
        )
        response_json = response.json()
        reply = response_json.get("response", "I couldn't get a response.")
    except Exception as e:
        print("Error connecting to server:", e)
        lessac.say("I'm having trouble connecting to the server.")
        continue

    print("Response from server:", reply)
    lessac.say(reply)
