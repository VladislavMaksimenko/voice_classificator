import wave
import struct
import time
import os
from pydub import AudioSegment
import contextlib
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from os import walk

def get_loud_times(wav_path, threshold=10000, time_constant=0.1):
    '''Work out which parts of a WAV file are loud.
        - threshold: the variance threshold that is considered loud
        - time_constant: the approximate reaction time in seconds'''

    wav = wave.open(wav_path, 'r')
    length = wav.getnframes()
    samplerate = wav.getframerate()

    assert wav.getnchannels() == 1, 'wav must be mono'
    assert wav.getsampwidth() == 2, 'wav must be 16-bit'

    # Our result will be a list of (time, is_loud) giving the times when
    # when the audio switches from loud to quiet and back.
    is_loud = False
    result = [(0., is_loud)]

    # The following values track the mean and variance of the signal.
    # When the variance is large, the audio is loud.
    mean = 0
    variance = 0

    # If alpha is small, mean and variance change slower but are less noisy.
    alpha = 1 / (time_constant * float(samplerate))

    for i in range(length):
        sample_time = float(i) / samplerate
        sample = struct.unpack('<h', wav.readframes(1))[0]
        #print(str(sample))

        # mean is the average value of sample
        mean = (1-alpha) * mean + alpha * sample

        # variance is the average value of (sample - mean) ** 2
        variance = (1-alpha) * variance + alpha * (sample - mean) ** 2
        #print(variance)
        # check if we're loud, and record the time if this changes
        new_is_loud = variance > threshold
        if is_loud != new_is_loud:
            result.append((sample_time, new_is_loud))
        is_loud = new_is_loud

    return result

def cut_voice(wav, volume_threshold, quant, temp):
    begin = time.time()
    if not os.path.exists(temp + "/chunks"):
        os.makedirs(temp + "/chunks")
    # Преобразование в одноканальный .wav
    newAudio = AudioSegment.from_wav(wav)
    newAudio = newAudio.set_channels(1)
    newAudio.export(temp + '/mono.wav', format="wav")
    # Получение списка отрезков, на которых присутствует голос
    loud_times = get_loud_times(temp + '/mono.wav', volume_threshold)
    # Рассчет длительности
    duration = 0
    with contextlib.closing(wave.open(wav,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = int(round(frames / float(rate))) - 1
    if duration > int(quant) * 1:
        duration = int(quant) * 1
    elif duration < 1:
        print('[ERR] Длительность звукового файла должна быть не менее 1 секунд')
        raise Exception()
    # Обрезка по полученным отрезкам
    all_time = 0
    t0 = False
    t = False
    chunk_id = 0
    for lt in loud_times:
        if lt[1] == True:
            t0 = lt[0] * 1000
        if lt[1] == False and t0 != False:
            t = lt[0] * 1000
            all_time = all_time + (t/1000 - t0/1000)
            if (all_time - 1) > duration:
                t = t - ((all_time - 1 - duration) * 1000)
                newChunk = AudioSegment.from_wav(temp + '/mono.wav')
                newChunk = newChunk[t0:t]
                newChunk = newChunk.set_channels(1)
                newChunk.export(temp + '/chunks/' + str(chunk_id) + '.wav', format="wav")
                break
            newChunk = AudioSegment.from_wav(temp + '/mono.wav')
            newChunk = newChunk[t0:t]
            newChunk = newChunk.set_channels(1)
            newChunk.export(temp + '/chunks/' + str(chunk_id) + '.wav', format="wav")
            t0 = False
            t = False
            chunk_id = chunk_id + 1
    if isinstance(t0, float) and t==False:
        newChunk = AudioSegment.from_wav(temp + '/mono.wav')
        newChunk = newChunk[t0:duration*1000]
        newChunk = newChunk.set_channels(1)
        newChunk.export(temp + '/chunks/' + str(chunk_id) + '.wav', format="wav")
    # Склеивание отрезков
    chunks = os.listdir(temp + '/chunks')
    merged_sound = False
    for chunk in chunks:
        if merged_sound != False:
            try:
                next_chunk = AudioSegment.from_wav(temp + '/chunks/' + chunk) #bug
                merged_sound = merged_sound + next_chunk
            except Exception as e:
                print('  ')
        else:
            try:
                merged_sound = AudioSegment.from_wav(temp + '/chunks/' + chunk)
            except Exception as e:
                print('  ')
    if merged_sound != False:
        merged_sound = merged_sound.set_channels(1)
        merged_sound.export(temp + '/merged.wav', format="wav")
    else:
        print('[INFO] Речь не найдена')
    for chunk in chunks:
        os.remove(temp + '/chunks/' + chunk)
    print('[INFO] Время удаления промежутков без речи: ' + str(time.time() - begin))

def makeChunks(wav, quant, temp):
    if not os.path.exists(temp + "/chunks_new"):
        os.makedirs(temp + "/chunks_new")
    begin = time.time()
    # Рассчет длительности
    duration = 0
    with contextlib.closing(wave.open(wav,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = int(round(frames / float(rate))) - 1
    print('[INFO] Длительность предобработанного звукового файла: ' + str(duration))
    if duration > int(quant) * 5:
        duration = int(quant) * 5 + 1
    elif duration < 1:
        print('[ERR] Длительность предобработанного звукового файла должна быть не менее 1 секунд')
        raise Exception()
    # Нарезка 5 секундных отрезков
    for i in range(1, duration+1, 1):  
        t1 = (i-1) * 1000 #Works in milliseconds
        t2 = i * 1000
        newAudio = AudioSegment.from_wav(wav)
        newAudio = newAudio[t1:t2]
        newAudio = newAudio.set_channels(1)
        newAudio.export(temp + '/chunks_new/' + wav.split('/')[-1].split('.')[0] + '_' + str(i) + str(time.time()) + '.wav', format="wav") 
    print('[INFO] Время нарезки фрагментов: ' + str(time.time() - begin))

def makePlots(dir, temp):
    if not os.path.exists(temp + "/plots"):
        os.makedirs(temp + "/plots")
    wavs = []
    for (_,_,filenames) in walk(temp + '/chunks_new'):
        wavs.extend(filenames)
        break
    for wav in wavs:
        input_data = read(temp + "/chunks_new/" + wav)
        audio = input_data[1]
        plt.plot(audio)
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.savefig(temp + "/plots/" + wav.split('.')[0] + '.png')
        plt.close('all')


