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
    print()
    generate_test()
    setup('convertFiles')

    file = 'Tchaik_F_144.wav'
    path = 'convertFiles/convertFiles (wav)/' + file

    data, sampleRate = librosa.load(path, sr=5000)
    audio = convert_to_audio(data)
    windowLength = 0.1
    interFrameTime = 0.01
    print("sample rate:", sampleRate)

    freqWithTime = np.abs(librosa.stft(audio,
                                       n_fft=int(windowLength*sampleRate),
                                       hop_length=int(interFrameTime*sampleRate)))
    print("stft dimensions (f, t):", freqWithTime.shape)

    f = np.linspace(0, sampleRate/2, freqWithTime.shape[0])
    t = np.linspace(0, interFrameTime*freqWithTime.shape[1], freqWithTime.shape[1])
    F, T = np.meshgrid(f, t)
    freqWithTime = freqWithTime.T

    energy = []
    for freq in range(freqWithTime.shape[0]):
        energy.append(np.mean(freqWithTime[freq]))
        freqWithTime[freq] = np.square(freqWithTime[freq])
        freqWithTime[freq] = freqWithTime[freq]/np.mean(freqWithTime[freq])

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(F, T, freqWithTime, cmap='viridis', alpha=0.8)
    plt.show()

    matchingArea = []
    for time in range(freqWithTime.shape[0] - 1):
        matchingArea.append(np.mean(np.maximum(freqWithTime[time + 1], freqWithTime[time])))
    matchingArea = np.array(matchingArea)
    matchingArea = matchingArea - np.mean(matchingArea)

    matchingFreq = np.array_split(np.fft.fft(matchingArea), 2)[0]
    beatFrequency = np.square(np.abs(matchingFreq))
    beatFramesFrequency = np.argmax(beatFrequency)/matchingFreq.shape[0]/2

    beatFrequency = beatFramesFrequency/interFrameTime
    print("tempo:", 60*beatFrequency)

    # ///////////////////////////////////////////////////////////////

    peaks = scipy.signal.find_peaks(matchingArea, prominence = 0.15)
    print(peaks[0])

    plt.plot(t[np.arange(matchingArea.shape[0])], matchingArea, 'b-')
    plt.vlines(t[peaks[0]], np.min(matchingArea), np.max(matchingArea), color='r', linestyles='dashed')

    energy = energy/np.mean(energy)
    energyPeaks = scipy.signal.find_peaks(energy, prominence = 1)[0]
    phase = t[energyPeaks[0]]

    lines = np.arange(0, t[matchingArea.shape[0]], 1/beatFrequency) + phase
    plt.vlines(lines, np.min(matchingArea), np.max(matchingArea), color='g', linestyles='dotted')

    plt.show()

    # //////////////////////////////////////////////////////

    I = matchingArea*np.sin(2*np.pi*beatFrequency*interFrameTime*np.linspace(0, matchingArea.shape[0], matchingArea.shape[0]))
    Q = matchingArea*np.cos(2*np.pi*beatFrequency*interFrameTime*np.linspace(0, matchingArea.shape[0], matchingArea.shape[0]))

    fig, ax = plt.subplots(4)
    ax[0].plot(np.sin(2*np.pi*beatFrequency*interFrameTime*np.linspace(0, matchingArea.shape[0], matchingArea.shape[0])))
    ax[1].plot(I)
    ax[2].plot(Q)
    ax[3].plot(matchingArea)
    plt.show()

    samples = I + 1j*Q
    importantSamples = np.arange(energyPeaks[0], matchingArea.shape[0], 1/beatFramesFrequency)
    samples = samples[importantSamples.astype(np.int32)]
    plt.plot(np.real(samples), np.imag(samples), '.')
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