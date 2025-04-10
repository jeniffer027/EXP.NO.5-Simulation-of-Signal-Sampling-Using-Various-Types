EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as i) Ideal Sampling ii) Natural Sampling iii) Flat Top Sampling

AIM
To simulate signal sampling using three different methods:

(i) Impulse Sampling

(ii) Natural Sampling

(iii) Flat-Top Sampling


SOFTWARE REQUIRED

Python with libraries: NumPy, Matplotlib
MATLAB


ALGORITHMS

Define the input signal x(t) – typically a sine wave or any continuous signal.

Choose sampling interval Ts – determines how often sampling occurs.

Create time vector t – fine-resolution time array for plotting.

Generate impulse train – set value to 1 at intervals of Ts, and 0 elsewhere.

Multiply x(t) with the impulse train – gives the impulse sampled signal.

Plot the original and sampled signals for comparison

PROGRAM

IMPULSE SAMPLING

#Impulse Sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
fs = 100
t = np.arange(0, 1, 1/fs) 
f = 5
signal = np.sin(2 * np.pi * f * t)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
NATURAL SAMPLING
#Natural sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# Parameters
fs = 1000  # Sampling frequency (samples per second)
T = 1  # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector
# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)
# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)
# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
pulse_train[i:i+pulse_width] = 1
# Natural Sampling
nat_signal = message_signal * pulse_train
# Reconstruction (Demodulation) Process
sampled_signal = nat_signal[pulse_train == 1]
# Create a time vector for the sampled points
sample_times = t[pulse_train == 1]
# Interpolation - Zero-Order Hold (just for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
# Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist
b, a = butter(order, normal_cutoff, btype='low', analog=False)
return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)
plt.figure(figsize=(14, 10))
# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)
# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
FLAT TOP SAMPLING
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# Parameters
fs = 1000               # Continuous-time sampling rate
f = 5                   # Message signal frequency
T = 1                   # Total time in seconds
ts = 1 / fs             # Time step
t = np.arange(0, T, ts) # Time vector

# Message signal
x = np.sin(2 * np.pi * f * t)

# Flat-top sampling parameters
fs_sampled = 50               # Sampling frequency
Ts = 1 / fs_sampled           # Sampling interval
tau = Ts / 4                  # Pulse width for flat-top
samples_idx = np.arange(0, len(t), int(Ts / ts))  # Sample indices

# Create pulse train
pulse_train = np.zeros_like(t)
for idx in samples_idx:
    pulse_train[idx:idx + int(tau / ts)] = 1

# Flat-top sampled signal (multiply signal with pulse train)
flat_top = x * pulse_train

# Reconstruct signal using resampling (just for visualization)
num_points = len(t)
reconstructed = resample(flat_top, num_points)

# Plotting
plt.figure(figsize=(12, 10))

# 1. Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, x, label='Original Message Signal')
plt.title('Original Message Signal')
plt.grid(True)
plt.legend()

# 2. Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.title('Pulse Train')
plt.grid(True)
plt.legend()

# 3. Flat-Top Sampled Signal
plt.subplot(4, 1, 3)
plt.plot(t, flat_top, color='orange', label='Flat-Top Sampled Signal')
plt.title('Flat-Top Sampled Signal')
plt.grid(True)
plt.legend()

# 4. Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed, color='green', label='Reconstructed Signal')
plt.title('Reconstructed Signal')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

OUTPUT

IDEAL SAMPLING

![image](https://github.com/user-attachments/assets/402de537-cb47-4cf4-84f0-b7366d02dab3)

NATURAL SAMPLING

![image](https://github.com/user-attachments/assets/00603af2-113c-47a7-b597-0d5e1ce4c0c1)

FLAT TOP SAMPLING

![image](https://github.com/user-attachments/assets/97a2ce04-063c-4120-a6ce-be58b80bd9b8)

RESULT

The simulation shows how three types of signal sampling work:

Impulse Sampling takes exact values at specific time points. It’s ideal but not practical in real life.

Natural Sampling uses short pulses to represent the signal. It’s more realistic but may change the shape a little.

Flat-Top Sampling holds each sampled value steady for a short time. This method is commonly used in real devices like ADCs.


