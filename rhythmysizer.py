import numpy as np
import librosa
import ffmpeg
import os
import matplotlib.pyplot as plt
import wave
import scipy.signal
import scipy.interpolate

def setup(folder: str) -> None:
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = folder + '/' + file
            newPath = folder + '/' + folder + ' (wav)/' + file[:-4] + '.wav'
            if not file.startswith('.') and not os.path.isfile(newPath):
                ffmpeg.input(path).output(newPath, loglevel='quiet', preset='ultrafast').run(overwrite_output=1)

def convert_to_audio(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        audio = np.add(data[:, 0], data[:, 1])
    else:
        audio = data
    # audio = nr.reduce_noise(y=audio, sr=sampleRate)
    return audio

def generate_test():
    t = np.linspace(0, 5, 5*22000)
    sampleRate = 22000
    frequency = 440*np.power(2, np.square(t/2).astype(np.int32)/12.0)
    data = np.sin(2*np.pi*frequency*t)
    frequency = 440*np.power(2, t.astype(np.int32)/12.0)
    data2 = np.sin(2*np.pi*frequency*t)
    data = np.append(data2, data)
    data = np.append(np.append(data2, data2), data)

    left_channel = data
    right_channel = data
    audio = np.array([left_channel, right_channel]).T
    audio = (audio * (2 ** 15 - 1)).astype("<h")

    with wave.open("convertFiles/test.wav", "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sampleRate)
        f.writeframes(audio.tobytes())

# ----------------------------------------------------------------------- #
# /////////////////////////////////////////////////////////////////////// #
# ----------------------------------------------------------------------- #

if __name__ == '__main__':
    # setup
    print()
    generate_test()
    setup('convertFiles')

    file = 'westside_beginning.wav'
    path = 'convertFiles/convertFiles (wav)/' + file

    data, sampleRate = librosa.load(path, sr=4000)
    audio = convert_to_audio(data)
    windowLength = 0.1
    interFrameTime = 0.025
    print("sample rate:", sampleRate)

    # spectrogram
    spectrogram = np.abs(librosa.stft(audio,
                                       n_fft=int(windowLength*sampleRate),
                                       hop_length=int(interFrameTime*sampleRate)))
    print("stft dimensions (f, t):", spectrogram.shape)

    f = np.linspace(0, sampleRate/2, spectrogram.shape[0])
    t = np.linspace(0, interFrameTime*spectrogram.shape[1], spectrogram.shape[1])
    F, T = np.meshgrid(f, t)
    spectrogram = spectrogram.T

    for freq in range(spectrogram.shape[0]):
        spectrogram[freq] = np.square(spectrogram[freq])
        spectrogram[freq] = spectrogram[freq]/np.mean(spectrogram[freq])

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(F, T, spectrogram, cmap='viridis', alpha=0.8)
    plt.show()

    # overlap
    spectralOverlap = []
    for time in range(spectrogram.shape[0] - 1):
        spectralOverlap.append(np.mean(np.maximum(spectrogram[time + 1], spectrogram[time])))
    spectralOverlap = np.array(spectralOverlap)

    overlapFrequencies = np.array_split(np.fft.fft(spectralOverlap), 2)[0]
    overlapFrequencies = np.square(np.abs(overlapFrequencies))
    excludeDC = 48
    excludeDC = int(excludeDC/60*interFrameTime*overlapFrequencies.shape[0]*2)

    tempo_fps = (np.argmax(overlapFrequencies[excludeDC:]) + excludeDC)/overlapFrequencies.shape[0]/2
    tempo_hz = tempo_fps/interFrameTime
    tempo_bpm = tempo_hz*60
    interbeat_time = 1/tempo_hz
    interbeat_frames = 1/tempo_fps

    plt.plot(np.abs(overlapFrequencies[1:]))
    plt.vlines(excludeDC, np.min(np.abs(overlapFrequencies)), np.max(np.abs(overlapFrequencies[1:])), color='r', linestyles="dashed")
    plt.show()
    print("tempo:", tempo_bpm)

    # beat matching
    beats_peak_derived = scipy.signal.find_peaks(spectralOverlap, prominence = 0.15)[0]

    pulses = np.zeros(spectralOverlap.shape[0])
    pulses[np.arange(0, spectralOverlap.shape[0] - 1, interbeat_frames).astype(np.int16)] = 1
    beat_sync = np.correlate(np.append(spectralOverlap, np.zeros(int(2*interbeat_frames))), pulses)
    initial_beat = scipy.signal.find_peaks(beat_sync, prominence = 1)[0][0]
    beats_tempo_calculated = np.arange(initial_beat, spectralOverlap.shape[0] - 1, interbeat_frames)
    beats_tempo_calculated = beats_tempo_calculated.astype(np.int16)

    plt.plot(t[np.arange(spectralOverlap.shape[0])], spectralOverlap, 'b-')
    plt.vlines(t[beats_peak_derived], np.min(spectralOverlap), np.max(spectralOverlap), color='r', linestyles='dashed')
    plt.vlines(t[beats_tempo_calculated], np.min(spectralOverlap), np.max(spectralOverlap), color='g', linestyles='dotted')
    plt.show()

    # fine beat synchronization
    alpha = 0.2
    correlation_vector = [1, 2, 1]
    true_beat = initial_beat + 2*interbeat_frames
    beats_loop_corrected = []
    interbeat_frame_log = []
    while true_beat < spectralOverlap.shape[0] - 2*interbeat_frames:
        interbeat_frame_log.append(interbeat_frames)
        beats_loop_corrected.append(true_beat)

        pulses = np.zeros(int(3*interbeat_frames))
        pulses[np.arange(interbeat_frames/2, pulses.shape[0], interbeat_frames).astype(np.int16)] = correlation_vector
        beat_sync = np.correlate(spectralOverlap[int(beats_loop_corrected[-1] - 2*interbeat_frames):
                                                  int(beats_loop_corrected[-1] + 2*interbeat_frames)],
                                                  pulses)
        correction = np.argmax(beat_sync) - 0.5*interbeat_frames

        interbeat_frames = interbeat_frames + alpha*correction
        true_beat = true_beat + interbeat_frames

    beats_loop_corrected = np.array(beats_loop_corrected).astype(np.int32)
    plt.plot(t[np.arange(spectralOverlap.shape[0])], spectralOverlap, 'b-')
    plt.vlines(t[beats_peak_derived], np.min(spectralOverlap), np.max(spectralOverlap), color='r', linestyles='dashed')
    plt.vlines(t[beats_loop_corrected], np.min(spectralOverlap), np.max(spectralOverlap), color='g', linestyles='dotted')
    plt.show()

    interbeat_frame_log = np.array(interbeat_frame_log)
    timeVariantTempo = 60/(interbeat_frame_log*interFrameTime)
    plt.plot(timeVariantTempo)
    plt.show()

    print()