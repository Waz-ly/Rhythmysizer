import numpy as np
import librosa
import ffmpeg
import os
import matplotlib.pyplot as plt
import wave

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
    frequency = 880*np.power(2, -(4*t).astype(np.int32)/12.0)
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
    generate_test()
    setup('convertFiles')

    file = 'violin_scale.wav'
    path = 'convertFiles/convertFiles (wav)/' + file

    data, sampleRate = librosa.load(path, sr=5000)
    audio = convert_to_audio(data)
    print(sampleRate)

    freqWithTime = np.abs(librosa.stft(audio, n_fft=int(0.1*sampleRate), center=False))
    # freqWithTime = freqWithTime/np.mean(freqWithTime)
    print(freqWithTime.shape)

    x = np.arange(freqWithTime.shape[0])
    y = np.arange(freqWithTime.shape[1])
    X, Y = np.meshgrid(x, y)
    freqWithTime = freqWithTime.T

    for freq in range(freqWithTime.shape[0]):
        freqWithTime[freq] = freqWithTime[freq]/np.mean(freqWithTime[freq])

    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, freqWithTime, cmap='viridis', alpha=0.8)
    plt.show()

    area = []
    for time in range(freqWithTime.shape[0] - 1):
        area.append(np.mean(np.minimum(freqWithTime[time + 1], freqWithTime[time])))
    
    plt.plot(area)
    plt.show()