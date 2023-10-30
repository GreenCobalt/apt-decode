import numpy as np
import scipy.io.wavfile
from scipy import spatial
import cv2, math, time, sys
from collections import deque
from time import sleep
from PIL import Image as im
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisignal
from scipy.signal import find_peaks, peak_widths

def cosine_similarity(a, b):
    return abs(1 - spatial.distance.cosine(a, b))

    if (all(v == 0 for v in a) or all(v == 0 for v in b)):
        return 0
    return sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))

def hilbert(data):
    print("Doing Hilbert Transform")
    analytical_signal = scisignal.hilbert(data)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope
    
def digitize(data):
    print("Digitizing data")
    data = np.round(255 * data / data.max())
    res = []
    for p in data:
        res.append(p[0])
    return np.array(res).astype(np.uint8)

def chunks(lst, n):
    res = []
    for i in range(0, len(lst), n):
        res.append(lst[i:i + n])
    res.pop()
    return res
    
def shift(arr, times):
    items = deque(arr)
    items.rotate(times)
    return list(items)
    
def find_nearest(value, array):
    pos = np.abs(array - value).argmin()
    return array[pos]
    
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
    
def findarrinarr(arr, find, thresh = 0.75):
    search = []
    for i in arr[0 : len(find)]:
        search.append(find_nearest(i, [0, 100]))
    for i in range(len(arr)):
        search.append(find_nearest(arr[i], [0, 100]))
        search.pop(0)
        
        image = np.array([search]).astype(np.uint8)
        height, width = image.shape
        resized = cv2.resize(image, (int(width * 20), int(height * 20)), interpolation = cv2.INTER_AREA)
        resized = cv2.putText(resized, str(round(cosine_similarity(search, find),2)), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', resized)
        cv2.waitKey(1)
        
        #if (i > frame_width * 1):
        #    return 0
            
        if (cosine_similarity(search, find) > thresh):
            if (i > (frame_width * 2) - 100):
                return findarrinarr(arr, find, thresh - 0.01)
            return i
        i += 4
    return -1

def doppler_align(signal, search):
    offset = 175
    peaks, _ = find_peaks(signal, distance=5000, height=200)
    
    img = []
    runAvg = []
    lastpeak = 0
    counter = 0
    
    for p in peaks:
        runAvg.append(p)
        if len(runAvg) > 4:
            runAvg.pop(0)
    
        pAvg = int(round(sum(runAvg) / len(runAvg), 0))
        img.append(signal[pAvg - offset : pAvg + 5512 - offset])
        
    img.pop()
    return img
    
    signal = sigii
    
    sigThresh = 6.7
    skipAfterDetect = 4000
    minPeak = 6.9
    fIndex = -1

    sims = []
    
    currLook = []
    
    for i in range(len(search)):
        currLook.append(find_nearest(signal[i], [0,100]))
        
    i = 0
    while i < int(round(len(signal) * 1, 0)) - 5512:
        if sum(currLook) == 0:
            sim = 0
        else:
            sim = cosine_similarity(currLook, search) * 10
        sims.append(sim)
        if (sim > sigThresh):
            #acc.append(currLook)
            i += skipAfterDetect
            for ii in range(skipAfterDetect):
                sims.append(0)
            if fIndex == -1:
                fIndex = i
            currLook = []
            for ii in range(i, i + len(search)):
                currLook.append(find_nearest(signal[ii], [0,100]))
        currLook.append(find_nearest(signal[i + 1], [0,100]))
        currLook.pop(0)
        i += 1
        
        #image = np.array([currLook]).astype(np.uint8)
        #height, width = image.shape
        #resized = cv2.resize(image, (int(width * 20), int(height * 20)), interpolation = cv2.INTER_AREA)
        #resized = cv2.putText(resized, str(round(sim,2)), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        #cv2.imshow('frame', resized)
        #cv2.waitKey(1)
        
        if (i % 1000 == 0):
            print(f"\r{round((i / len(signal))*100, 2)}%", flush=True, end="")
    print()
    #print(acc)
    
    #sims = sims[fIndex : len(sims)]
    peaks, _ = find_peaks(sims, distance=5000)
    
    img = []
    i = 0
    for p in peaks:
        if (sims[p] < minPeak):
            if ((i+1) == len(peaks)):
                peaks[i] = peaks[i-1]
            else:
                peaks[i] = (peaks[i+1] + peaks[i-1]) / 2 
        img.append(signal[p : p + 5512])
        i += 1
    
    return img

def correct_doppler(signal, correctdoppler):
    syncA = [0,0,0,0,100,100,100,0,0,0,100,100,100,0,0,0,100,100,100,0,0,0,100,100,100,0,0,0,100,100,100,0,0,100,100,100,0,0,0,100,100,100,0,0,0]
    
    if (correctdoppler):
        print("Correcting for Doppler shift")
        return doppler_align(signal, syncA)
    else:
        arr = []
        for i in range(0, len(signal), frame_width):
            arr.append(signal[i : i + frame_width])
        arr.pop()
        return arr
    
print("Reading WAV")
(rate, raw_signal) = scipy.io.wavfile.read(sys.argv[1])
frame_width = int(0.5*rate)
filtered_signal = digitize(hilbert(raw_signal))

signals = [(False, "normal"), (True, "doppler")]

for s in signals:
    signal = correct_doppler(filtered_signal, s[0])
    arr = np.array(signal)
    image = im.fromarray(arr)
    image = ImageOps.flip(image)
    print(f'Saving out_{s[1]}.png')
    image.save(f'out_{s[1]}.png')