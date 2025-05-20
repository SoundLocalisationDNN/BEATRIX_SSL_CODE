import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd
from scipy.signal import butter, filtfilt, resample_poly
import matplotlib.pyplot as plt
import serial
from matplotlib.colors import Normalize

#Import required functions (in your working directory)
from Gammatone_spectogram_Function import compute_gammatone_spectrogram
from GCCPHAT_ZeroPad_Freqrange import gcc_phat

#Directory to save plots
SAVE_DIR = "/path/to/save/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

#global counter
plot_counter = 0

def send_command(ser, command):
    """Send a command to the Arduino and read a response (if any)."""
    ser.write(command.encode())
    print("Sent:", repr(command))
    time.sleep(0.1)  #Allow Arduino time to process
    response = ser.readline().decode().strip()
    if response:
        print("Received:", response)
    else:
        response = ""
    return response

# Bandpass Filter Function (using order=3 as in training)
# =============================================================================
def bandpass_filter(data, sample_rate, lowcut=50, highcut=8000, order=3):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


# Neural Network Model
# =============================================================================
class AudioRegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec_conv = nn.Sequential(
            #Block 1
            nn.Conv2d(3,  64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # downsample

            #Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            #Block 3
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # GCC branch
        self.gcc_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Conv1d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2,2),
        )

         #Fusion
        fusion_dim = 256 + 128  
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, spec_left, spec_right, gcc):
        diff = spec_left - spec_right
        x = torch.cat([spec_left, spec_right, diff], dim=1)  
        h = self.spec_conv(x)                               
        f_spec = h.mean(dim=[2,3])                          

        g = self.gcc_conv(gcc)                             
        f_gcc = g.mean(dim=2)                               

        out = torch.cat([f_spec, f_gcc], dim=1)              


#Continuous Recording Setup using a Ring Buffer
# =============================================================================
fs = 48000       # Sampling rate (Hz)
duration = 0.5 
channels = 2  
device_index = 1  #Adjust
buffer_size = int(fs * duration)

# Initialize a ring buffer with zeros
ring_buffer = np.zeros((buffer_size, channels), dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    global ring_buffer
    if status:
        print(status)
    # Roll the ring buffer to make room for new data and add the new frames at the end.
    ring_buffer = np.roll(ring_buffer, -frames, axis=0)
    ring_buffer[-frames:] = indata


# Detection and Processing Functions
# =============================================================================
def detect_audio(segment, threshold=0.01):
    """Return True if the Mean Absolute Value exceeds the threshold."""
    mav = np.mean(np.abs(segment))
    print(f"Segment MAV: {mav:.6f}")
    return mav > threshold

def process_audio_segment(segment, fs, model, device, norm_params):
    """
    Pre-process the segment similarly to training:
        Compute gammatone spectrogram and GCC-PHAT features,
        then normalise these features using saved parameters.
    """

    normalized_segment = segment

    #Parameters for spectrogram computation
    window_time = 0.1 
    hop_time =  window_time*0.5   
    fmin = 50
    fmax = 8000

    #Compute gammatone spectrogram for each channel.
    specs, _ = compute_gammatone_spectrogram(normalized_segment, fs, fmin, fmax,
                                              window_time=window_time, hop_time=hop_time)
    #Compute GCC-PHAT features between channels.
    sig1 = normalized_segment[:, 0]
    sig2 = normalized_segment[:, 1]
    _, cc, _, _ = gcc_phat(sig1, sig2, fs)


    feat = {
        "gammatone_left": specs[0],
        "gammatone_right": specs[1],
        "gcc": cc
    }

    #Normalise features using saved parameters.
    merged_mean = norm_params['merged_mean']
    merged_std = norm_params['merged_std']
    gcc_mean = norm_params['gcc_mean']
    gcc_std = norm_params['gcc_std']
    
    g_left  = feat["gammatone_left"]
    g_right = feat["gammatone_right"]
    gcc     = feat["gcc"]

    feat["gammatone_left"] = (feat["gammatone_left"] - merged_mean) / (merged_std + 1e-8)
    feat["gammatone_right"] = (feat["gammatone_right"] - merged_mean) / (merged_std + 1e-8)
    feat["gcc"] = (feat["gcc"] - gcc_mean) / (gcc_std + 1e-8)
    


    #Convert features to PyTorch tensors
    spec_left = torch.tensor(feat["gammatone_left"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spec_right = torch.tensor(feat["gammatone_right"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gcc_feat = torch.tensor(feat["gcc"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    

    model.eval()
    with torch.no_grad():
        output = model(spec_left.to(device), spec_right.to(device), gcc_feat.to(device))
    pred = output.cpu().numpy()[0]
    pred_angle = math.degrees(math.atan2(pred[0], pred[1]))
    
    print(f"Predicted azimuth angle: {pred_angle:.2f}°\n")
    
    #Prepare axes data
    gcc_lag_ms = 0.65
    n_bands, n_frames = g_left.shape
    duration = segment.shape[0] / 44100  # or your sr variable
    # time edges for pcolormesh
    time_edges = np.linspace(0, duration, n_frames + 1)
    #frequency edges log‐spaced
    freq_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    lags_ms = np.linspace(-gcc_lag_ms, gcc_lag_ms, len(gcc))

    #Compute difference spectrogram
    g_diff = g_left - g_right

    vmin_lr = min(g_left.min(), g_right.min())
    vmax_lr = max(g_left.max(), g_right.max())
    norm_lr = Normalize(vmin=vmin_lr, vmax=vmax_lr)

    #Separate diff color scale
    vmin_diff = g_diff.min()
    vmax_diff = g_diff.max()
    norm_diff = Normalize(vmin=vmin_diff, vmax=vmax_diff)

    duration = segment.shape[0] / 44100  #or your sr
    n_bands, n_frames = g_left.shape
    time_edges = np.linspace(0, duration, n_frames + 1)
    freq_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    lags_ms = np.linspace(-gcc_lag_ms, gcc_lag_ms, len(gcc))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_left, ax_right = axs[0, 0], axs[0, 1]
    ax_diff, ax_gcc = axs[1, 0], axs[1, 1]

    #Left
    pcm0 = ax_left.pcolormesh(time_edges, freq_edges, g_left, shading='auto', norm=norm_lr)
    ax_left.set(title="Left Gammatone", xlabel="Time (s)", ylabel="Freq (Hz)")
    fig.colorbar(pcm0, ax=ax_left, pad=0.02).set_label("Amp")

    #Right
    pcm1 = ax_right.pcolormesh(time_edges, freq_edges, g_right, shading='auto', norm=norm_lr)
    ax_right.set(title="Right Gammatone", xlabel="Time (s)", ylabel="Freq (Hz)")
    fig.colorbar(pcm1, ax=ax_right, pad=0.02).set_label("Amp")

    #Difference
    pcm2 = ax_diff.pcolormesh(time_edges, freq_edges, g_diff, shading='auto', norm=norm_diff)
    ax_diff.set(title="Diff (Left − Right)", xlabel="Time (s)", ylabel="Freq (Hz)")
    fig.colorbar(pcm2, ax=ax_diff, pad=0.02).set_label("Δ Amp")

    #GCC‐PHAT
    ax_gcc.plot(lags_ms, gcc)
    ax_gcc.set(title="GCC-PHAT", xlabel="Lag (ms)", ylabel="Correlation")
    ax_gcc.grid(True)

    plt.tight_layout()

    #Save as before
    global plot_counter
    fname = os.path.join(
        SAVE_DIR,
        f"plot_{plot_counter:04d}_angle_{pred_angle:.1f}.png"
    )
    fig.savefig(fname)
    print(f"Saved plot to {fname}")
    plot_counter += 1

    #Optionally display it
    plt.show()
   
    
    return pred_angle


# Main Script: Continuous Audio Monitoring and Prediction
# ===========================================================================
def main():
    mav_threshold = 0.0005  # Adjust as needed
    conversion_factor = 11.66666
    move = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AudioRegressionCNN().to(device)
    checkpoint_path = r"/home/js3638/NeuralNetwork/CNN(100ms,dB,TIMIT)/sound_localisation_model_CCN.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    norm_params = checkpoint['feature_normalization']
    print("Model and normalization parameters loaded.\n")
    print(sd.query_devices())
    
    port = "/dev/ttyACM0"  #Change as needed.
    baud_rate = 9600
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print("Serial port opened.")
        time.sleep(2)
    except Exception as e:
        print("Error opening serial port:", e)
        return

    # Enable motors.
    print("Enabling motors...")
    send_command(ser, "@ENMOTORS ON\r")
    time.sleep(1)
    
    #Calibrate the robot to set its home position.
    print("Calibrating robot...")
    response = send_command(ser, "@CALNOW\r")
    if "NACK" in response:
        print("Calibration failed. Check that the robot is in its home position.")
        ser.close()
        return
    time.sleep(1)
    
    
    print("Starting continuous audio monitoring...\n")

    #Start the input stream with the callback for continuous recording.
    stream = sd.InputStream(samplerate=fs, device=device_index, channels=channels, callback=audio_callback)
    with stream:
        try:
            while True:
                #Copy the most recent 0.5 seconds from the ring buffer.
                current_segment = ring_buffer.copy()


                filtered_segment = resample_poly(bandpass_filter(current_segment, fs, order=3),44100,fs,axis = 0)
                
                #Detect if significant audio is present
                if detect_audio(filtered_segment, threshold=mav_threshold):
                    print("Audio detected. Processing segment...")
                    sd.play(filtered_segment,44100)
                    sd.wait()
                    max_int32 = np.iinfo(np.int32).max
                    int32_data = (filtered_segment * max_int32).astype(np.int32)  #Emulate 32-bit signed PCM (same as training data)
                    filtered_segment = int32_data
                    predicted_angle = process_audio_segment(filtered_segment, 44100, model, device, norm_params)
                    print(f"Predicted azimuth angle: {predicted_angle:.2f}°\n")
                    if move == True:
                        steps = int(-1 * predicted_angle * conversion_factor)
                        speed_value = 100
                        move_command = f"@MOVRALL {steps} {steps} {steps} 0 {speed_value} {speed_value} {speed_value} 0\r"
    
                        print(f"Turning head by {steps} steps using @MOVRALL...")
                        send_command(ser, move_command)
                        time.sleep(abs(steps) / speed_value)
                        print("Turn complete!")
                                        
                else:
                    print("No significant audio detected. Continuing to monitor...\n")

                #Process at a desired interval
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Audio monitoring stopped.")

if __name__ == '__main__':
    main()
