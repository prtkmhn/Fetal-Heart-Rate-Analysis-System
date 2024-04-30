from __future__ import print_function, division, unicode_literals
import wave
import numpy as np
import matplotlib.pyplot as plt

def smooth2(x, window_width):
    """Smooth data and reduce kicks by averaging within a window, preserving peaks."""
    count = 0
    data = np.array(x)
    for i in range(0, len(data), window_width):
        arr = data[i:(i + window_width - 1)]
        arr = np.absolute(arr)  # Take the magnitude
        avg = np.average(arr)
        maxi = max(arr)
        ind = i + np.argmax(arr)
        diff = maxi - avg
        if maxi > 110:
            count += 1
            for j in range(i, (i + window_width - 1)):
                if j != ind:
                    data[j] = avg
    print(count)
    return data

# Main function
if __name__ == "__main__":
    wr = wave.open('Audio.wav', 'r')
    par = list(wr.getparams())  # Get the parameters from the input.
    n = wr.getnframes()
    t = wr.getframerate()
    time = int(n / t)
    da = np.frombuffer(wr.readframes(time * t), dtype=np.int16)
    win = 0.025
    print(da)
    plt.figure(1)
    plt.plot(da)
    result1_smooth = smooth2(da, 1000000)
    plt.figure(2)
    plt.plot(result1_smooth)
    plt.show()