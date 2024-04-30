from __future__ import print_function, division, unicode_literals
import wave
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.stats import linregress

# Utility functions
def mean(numbers):
    """Calculate the mean of a list of numbers."""
    return float(sum(numbers)) / max(len(numbers), 1)

def percentChange(startPoint, currentPoint):
    """Calculate the percentage change between two points."""
    try:
        x = ((float(currentPoint) - startPoint) / abs(startPoint)) * 100.00
        return x if x != 0.0 else 0.000000001
    except:
        return 0.0001

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter."""
    from math import factorial
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # Precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # Pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def get_peaks(x, win):
    """Identify peaks in the data using a sliding window approach."""
    ind = win
    peaks_y = []
    peaks_x = []
    flag = False

    while ind < len(x):
        flag = False
        if ind + win < len(x):
            for i in range(1, win + 1):
                if x[ind] > x[ind - i] and x[ind] > x[ind + i]:
                    j = 0
                else:
                    flag = True
                    break
        else:
            for i in range(1, len(x) - ind):
                if x[ind] > x[ind - i] and x[ind] > x[ind + i]:
                    j = 0
                else:
                    flag = True
                    break
        if flag:
            ind += 1
        else:
            peaks_x.append(ind)
            peaks_y.append(x[ind])
            ind += win
    return peaks_x, peaks_y

def smooth2(x, window_width):
    """Smooth data using a simple moving average."""
    data = np.array(x)
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def removeOutliers(x, c):
    """Remove outliers from data based on standard deviation."""
    a = np.array(x)
    mean = np.mean(a)
    sd = np.std(a)
    resultList = [i if mean - c * sd <= i <= mean + c * sd else 0 for i in a.tolist()]
    return resultList

def find_fft(da, t, lowpass, highpass):
    """Find the Fast Fourier Transform of the data and apply a bandpass filter."""
    result = np.zeros_like(da)
    temp_fft = []
    temp_inverse = []
    mono = np.zeros(t)
    ind = 0
    while ind < len(da):
        mono[0:t // 2] = mono[t // 2:t]
        mono[t // 2:t] = da[ind:ind + (t // 2)]
        temp_fft = np.fft.rfft(mono)
        for i in range(0, len(temp_fft)):
            if i < lowpass or i > highpass:
                temp_fft[i] = 0
        temp_inverse = np.fft.irfft(temp_fft)
        result[ind: ind + (t // 2)] = temp_inverse[0: t // 2]
        ind += (t // 2)
    return result

def av_points(res, win, t):
    """Average points over a specified window."""
    l = len(res)
    result = []
    win_p = win * t
    it = math.ceil((l / t) / win)
    for i in range(int(it)):
        buf = res[int(win_p * i): int(win_p * (i + 1))]
        result.append(mean(buf))
    return result

# Main function
if __name__ == "__main__":
    wr = wave.open('Audio.wav', 'r')
    par = list(wr.getparams())  # Get the parameters from the input.
    n = wr.getnframes()
    t = wr.getframerate()
    time = int(n / t)
    lowpass = 1 * (t / (n / 2))  # Remove lower frequencies.
    highpass = 10 * (t / (n / 2))  # Remove higher frequencies.
    da = np.frombuffer(wr.readframes(time * t), dtype=np.int16)
    win = 0.025
    result = find_fft(da, t, lowpass, highpass)
    result = removeOutliers(result, 4)
    result1_smooth = smooth2(result, win * t)
    result1 = av_points(result1_smooth, win, t)
    result1_peaks_x, result1_peaks_y = get_peaks(result1, 10)
    plt.figure(1)
    plt.plot(np.arange(0, time, win), result, '-gD', markevery=result1_peaks_x)
    plt.figure(2)
    try:
        plt.plot(np.arange(0, time - win, 1 / t), result1_smooth[:])
    except:
        plt.plot(np.arange(0, time - win, 1 / t), result1_smooth[:-1])
    plt.show()
    wr.close()