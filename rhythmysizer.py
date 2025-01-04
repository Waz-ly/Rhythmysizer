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
    t = np.linspace(0, 3, 3*22000)
    sampleRate = 22000
    frequency = 440*np.power(2, 2*(4*t).astype(np.int32)/12.0)
    data = np.sin(2*np.pi*frequency*t)

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

    file = 'Tchaik_F_100.wav'
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
    spectralOverlap = spectralOverlap - np.mean(spectralOverlap)

    overlapFrequencies = np.array_split(np.fft.fft(spectralOverlap), 2)[0]
    overlapFrequencies = np.square(np.abs(overlapFrequencies))
    excludeDC = int(windowLength*sampleRate/2/50)

    tempo_fps = (np.argmax(overlapFrequencies[excludeDC:]) + excludeDC)/overlapFrequencies.shape[0]/2
    tempo_hz = tempo_fps/interFrameTime
    tempo_bpm = tempo_hz*60

    plt.plot(np.abs(overlapFrequencies))
    plt.show()

    print("tempo:", tempo_bpm)

    # beat matching
    beats_peak_derived = scipy.signal.find_peaks(spectralOverlap, prominence = 0.15)
    plt.plot(t[np.arange(spectralOverlap.shape[0])], spectralOverlap, 'b-')
    plt.vlines(t[beats_peak_derived[0]], np.min(spectralOverlap), np.max(spectralOverlap), color='r', linestyles='dashed')

    pulses = np.zeros(spectralOverlap.shape[0])
    pulses[np.arange(0, spectralOverlap.shape[0], 1/tempo_fps).astype(np.int16)] = 1
    beat_sync = np.correlate(np.append(spectralOverlap, np.zeros(int(2/tempo_fps))), pulses)
    initial_beat = scipy.signal.find_peaks(beat_sync, prominence = 1)[0][0] + 1/tempo_fps

    beats_tempo_calculated = np.arange(0, t[spectralOverlap.shape[0]], 1/tempo_hz) + t[int(initial_beat)]
    plt.vlines(beats_tempo_calculated, np.min(spectralOverlap), np.max(spectralOverlap), color='g', linestyles='dotted')

    plt.show()
    
    # I = overlapFrequencies*np.sin(2*np.pi*tempo_hz*interFrameTime*np.linspace(0, overlapFrequencies.shape[0], overlapFrequencies.shape[0]))
    # Q = overlapFrequencies*np.cos(2*np.pi*tempo_hz*interFrameTime*np.linspace(0, overlapFrequencies.shape[0], overlapFrequencies.shape[0]))

    # samples = I + 1j*Q
    # importantSamples = np.arange(0, overlapFrequencies.shape[0], 1/tempo_fps) # change
    # samples = samples[importantSamples.astype(np.int32)]
    # plt.plot(np.real(samples), np.imag(samples), '.')
    # plt.show()

    # print()

    # unsure if this works, more testing needed
    N = beats_tempo_calculated.shape[0]
    pulses = np.zeros(int(3/tempo_fps))
    pulses[np.arange(0, pulses.shape[0], 1/tempo_fps).astype(np.int16)] = 1
    alpha = 0.132
    beta = 0.00932
    freq = tempo_fps
    phase = 0
    beats_loop_corrected = []
    for i in range(N - 4):
        beats_loop_corrected.append(i/tempo_fps + initial_beat + phase)
        true_beats = np.correlate(spectralOverlap[int(beats_loop_corrected[i]-0.5/tempo_fps):int(beats_loop_corrected[i]+3.5/tempo_fps)], pulses)
        freq_correction = np.max(true_beats) - 0.5/tempo_fps
        freq = 1/(1/freq + alpha*freq_correction)
        phase = phase + beta/freq
    beats_loop_corrected = np.array(beats_loop_corrected).astype(np.int32)

    plt.plot(t[np.arange(spectralOverlap.shape[0])], spectralOverlap, 'b-')
    plt.vlines(t[beats_peak_derived[0]], np.min(spectralOverlap), np.max(spectralOverlap), color='r', linestyles='dashed')
    plt.vlines(t[beats_loop_corrected], np.min(spectralOverlap), np.max(spectralOverlap), color='g', linestyles='dotted')
    plt.show()

    # alternating_sequence = np.arange(0, peaks[0].shape[0])
    # alternating_sequence = np.power(-1, alternating_sequence)
    # plt.plot(t[peaks[0]], alternating_sequence)
    # spline = scipy.interpolate.CubicSpline(t[peaks[0]], alternating_sequence)

    # beats = spline(t)
    # plt.plot(t, beats)
    # plt.show()

    # beats = beats*scipy.signal.windows.hann(t.shape[0])
    # beatFrequencies = np.array_split(np.abs(np.fft.fft(beats)), 2)[0]
    # f = np.linspace(0, 1/(2*interFrameTime), beatFrequencies.shape[0])
    # plt.plot(f, beatFrequencies)
    # plt.show()

    # # times 2 because alternating sequence halves the frequency
    # peak = 2*scipy.signal.find_peaks(beatFrequencies, distance = beatFrequencies.shape[0])[0]
    # print(peak)
    # print("beat frequency (Hz):", f[peak[0]])
    # print("beat frequency (bpm):", f[peak[0]]*60)