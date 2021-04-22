from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile
import time
import sys
import preprocess
import traceback
import classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # less TF messages

def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape

def clean_temprorary(temp):
    chunks = []
    try:
        if os.path.exists(temp + '/chunks'):
            chunks = os.listdir(temp + '/chunks')
            for chunk in chunks:
                os.remove(temp + '/chunks/' + chunk)
            os.rmdir(temp + '/chunks')
        if os.path.exists(temp + '/chunks_new'):
            chunks = os.listdir(temp + '/chunks_new')
            for chunk in chunks:
                os.remove(temp + '/chunks_new/' + chunk)
            os.rmdir(temp + '/chunks_new')
        if os.path.exists(temp + '/plots'):
            plots = os.listdir(temp + '/plots')
            for plot in plots:
                os.remove(temp + '/plots/' + plot)
            os.rmdir(temp + '/plots')
        if os.path.exists(temp + '/merged.wav'):
            os.remove(temp + '/merged.wav')
        if os.path.exists(temp + '/mono.wav'):    
            os.remove(temp + '/mono.wav')
    except Exception as e:
        print('[WARNING] Не все временные файлы удалены.')

def processSingleWav(outDir, wav, model, pauseTime, mono, resample, frags, volume_threshold, temp):
    try:
        print('[INFO] Предобработка звука...')
        # Предобработка найденного файла
        try: 
            preprocess.cut_voice(wav, volume_threshold, frags, temp)
        except Exception as e:
            clean_temprorary(temp)
            json_file = open(outDir + '/' + wav.split('/')[-1] + ".json", "w")
            json_file.write('{\n\t"age":\t{')
            outstr = '\n\t"adult": -1'
            json_file.write(outstr)
            json_file.flush()  
            json_file.write("\n\t}\n}\n")
            json_file.close()
            print('[ERR] Ошибка при предобработке: ' + str(e))
            return False
        try:
            preprocess.makeChunks(temp + '/merged.wav', frags, temp)
        except Exception as e:
            clean_temprorary(temp)
            json_file = open(outDir + '/' + wav.split('/')[-1] + ".json", "w")
            json_file.write('{\n\t"age":\t{')
            outstr = '\n\t\t"adult": -1'
            json_file.write(outstr)
            json_file.flush()  
            json_file.write("\n\t}\n}\n")
            json_file.close()
            print('[ERR] Ошибка при создании отрезков: ' + str(e))
            return False
        t0 = time.time()
        chunks = os.listdir(temp + '/chunks_new')
        fact_frags = len(chunks)
        if fact_frags == 0:
            clean_temprorary(temp)
            return False
        print('[INFO] Выделено ' + str(fact_frags) + ' фрагментов для классификации голоса')
        #Преобразование звука в изображение спектра
        try:
            preprocess.makePlots(temp + '/chunks_new', temp)
        except Exception as e:
            clean_temprorary(temp)
            json_file = open(outDir + '/' + wav.split('/')[-1] + ".json", "w")
            json_file.write('{\n\t"age":\t{')
            outstr = '\n\t"adult": -1'
            json_file.write(outstr)
            json_file.flush()  
            json_file.write("\n\t}\n}\n")
            json_file.close()
            print('[ERR] Ошибка при создании изображений спектра: ' + str(e))
            return False
        print('[INFO] Изображения спектра созданы')
        print('[INFO] Классификация...')
        predictions = []
        plots = os.listdir(temp + '/plots')
        for plot in plots:
            #Обработка
            result = classification.processSingleImage(temp + '/plots/' + plot, model, 0.5)
            if result:
                predictions.append('adult')
            else:
                predictions.append('child')
        #json
        wav_name = wav
        if len(wav.split('/')) > 1:
            wav_name = wav.split('/')[len(wav.split('/'))-1]
        elif len(wav.split('\\')) > 1:
            wav_name = wav.split('\\')[len(wav.split('\\'))-1]
        json_file = open(outDir + '/' + wav_name + ".json", "w")
        json_file.write('{\n\t"age":\t{')
        # Рассчет вероятностей
        child_quant = 0
        adult_quant = 0
        for pred in predictions:
            if pred == 'adult':
                adult_quant = adult_quant + 1
            else:
                child_quant = child_quant + 1
        child_pos = child_quant / fact_frags
        adult_pos = adult_quant / fact_frags
        if child_pos > adult_pos:
            print("[INFO] Результат классификации: 'ребёнок' " + str(child_pos * 100) + '%')
        else:
            print("[INFO] Результат классификации: 'взрослый' " + str(adult_pos * 100) + '%')
        outstr = '\n\t"adult": ' + str(adult_pos*100) 
        json_file.write(outstr)
        json_file.flush()  
        json_file.write("\n\t}\n}\n")
        json_file.close()
        # clean temprorary
        clean_temprorary(temp)
        t_end = time.time() - t0
        print('[INFO] Время классификации: ' + str(t_end))
        return True
    except Exception as e:
        print('[ERR] Ошибка при обработке .wav ' + str(e))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback)
        time.sleep(1)
        clean_temprorary(temp)
        return False


def processLoop(inDir, outDir, model, pauseTime, mono, resample, frags, volume_threshold, temp, clear_time):
    loop_mode_db = []
    attempts = 1
    attempt_files = []
    timer_ = time.time()
    while True:
        if (time.time() - timer_) > clear_time:
            loop_mode_db = []
            timer_ = time.time()
        clean_temprorary(temp)
        files = os.listdir(inDir)
        wavs = list(filter(lambda x: x.endswith('.wav'), files))
        for wav in wavs:
            counter = 0
            for item in loop_mode_db:
                if wav == item:
                    counter = counter + 1
            if counter == 0:
                print('\nНайден новый .wav файл: ' + str(wav) + '\n')
                try:
                    processSingleWav(outDir, inDir + '/' + wav, model, pauseTime, mono, resample, frags, volume_threshold, temp)
                    loop_mode_db.append(wav)
                except PermissionError:
                    print('[INFO] Файл еще загружается или к нему нет доступа... Попытка: ' + str(attempts) + ' из 5.')
                    if wav in attempt_files:
                        attempts = attempts + 1
                        if attempts > 5:
                            print('[INFO] Файлы: ' + str(attempt_files) + ' не удалось обработать. Они будут проигнорированы.')
                            for af in attempt_files:
                                loop_mode_db.append(af)
                            attempts = 1
                            attempt_files = []
                    else:
                        attempt_files.append(wav)
                    continue
                except RuntimeWarning:
                    print('[INFO] Файл занят другим процессом...')
                    continue
        time.sleep(pauseTime)


def main(args):
    # Read args
    print('[INFO] Reading arguments')
    """
    log_name = 'log_' + str(time.time()) + '.txt'
    log_file = open(log_name, 'w')
    log_file.write('Reading arguments: ') 
    log_file.close()
    """
    weights_file=args.weights
    """
    log_file = open(log_name, 'a')
    log_file.write('\n\tweights file: ' + str(weights_file)) 
    log_file.close()
    """
    resample = args.resample
    mono = args.mono
    in_dir = 'None'
    try:
        in_dir = args.dir
    except Exception as e:
        """
        log_file = open(log_name, 'a')
        log_file.write('\n[ERR] input dir reading error: ' + str(e)) 
        log_file.close()
        """
        os._exit(0)
    if in_dir == 'None':
        in_dir = os.getcwd()
    """
    log_file = open(log_name, 'a')
    log_file.write('\n\tinput dir: ' + str(in_dir)) 
    log_file.close()
    """
    out_dir = 'None'
    try:
        out_dir = args.output
    except Exception as e:
        """
        log_file = open(log_name, 'a')
        log_file.write('\n[ERR] output dir reading error: ' + str(e)) 
        log_file.close()
        """
        os._exit(0)
    """
    log_file = open(log_name, 'a')
    log_file.write('\n\toutput dir: ' + str(out_dir)) 
    log_file.close()
    """
    loop_timer = int(args.loop_timer)
    frags = args.fragments
    volume_threshold = float(args.volume_threshold)
    single_file_mode = False
    if args.single_file_mode == 'on':
        single_file_mode = True
    wav_path = None
    if single_file_mode:
        wav_path = args.wav_path
    if single_file_mode and wav_path == None:
        print('[ERR] В режиме обработки одного файла нужно указать файл --wav_path=')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    temp = args.temp_path
    if not os.path.exists(temp):
        os.makedirs(temp)
    clear_time = int(args.clear_time)

    # Load the model
    print('[INFO] Loading neural model and weights')
    model = None
    try:
        model = classification.loadModel(weights_file)
    except Exception:
        print("[ERR] No weights file found.  Aborting")
        """
        log_file = open(log_name, 'a')
        log_file.write('\n[ERR] No weights file found.  Aborting')
        log_file.close()
        """
        os._exit(0)
    if model is None:
        print("[ERR] No weights file found.  Aborting")
        """
        log_file = open(log_name, 'a')
        log_file.write('\n[ERR] No weights file found.  Aborting')
        log_file.close()
        """
        os._exit(0)
    """
    log_file = open(log_name, 'a')
    log_file.write('\n[INF] Neural model loaded')
    log_file.close()
    """

    # Processing wav in single mode:
    if single_file_mode:
        print('[INFO] Processing wav in single file mode\n')
        try:
            processSingleWav(out_dir, wav_path, model, loop_timer, mono, resample, frags, volume_threshold, temp)
        except PermissionError:
            """
            log_file = open(log_name, 'a')
            log_file.write('\n[WAR] File still loading or not accessible')
            """
            print('[INFO] Файл еще загружается или к нему нет доступа...')
        except RuntimeWarning:
            """
            log_file = open(log_name, 'a')
            log_file.write('\n[WAR] File using by another process')
            """
            print('[INFO] Файл занят другим процессом...')
        except Exception as e:
            """
            log_file = open(log_name, 'a')
            log_file.write('\n[ERR] Processing in single wave mode failed')
            log_file.close()
            """
            print('[ERR] Ошибка: \n' + str(e) + '\nЗавершение работы\n')
            os._exit(0)
    else:
    #Processing loop
        print('[INFO] Starting processing loop ("CTRL + C" for exit)\n')
        try:
            processLoop(in_dir, out_dir, model, loop_timer, mono, resample, frags, volume_threshold, temp, clear_time)
        except KeyboardInterrupt:
            print('[INFO] Завершение работы классификатора...')
            """
            log_file = open(log_name, 'a')
            log_file.write('\n[OK] Programm ended by keyboard interruption in loop mode')
            log_file.close()
            """
            os._exit(0)
        except Exception as e:
            """
            log_file = open(log_name, 'a')
            log_file.write('\n[ERR] Processing in loop mode failed: ' + str(e))
            log_file.close()
            """
            print('[ERR] Ошибка: \n' + str(e) + '\nЗавершение работы\n')
            os._exit(0)
    """
    log_file = open(log_name, 'a')
    log_file.write('\n[OK] Programm ended normally')
    log_file.close()
    """
    os._exit(0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="predicts which class file(s) belong(s) to")
    parser.add_argument('-w', '--weights', help='weights file in hdf5 format', default="weights.h5")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    parser.add_argument('-o', "--output", help="out dir", default='output')
    parser.add_argument('-i', "--dir", help="processing dir", default='None', required=False)
    parser.add_argument('-lt', "--loop_timer", help="loop timer", default=10, required=False)
    parser.add_argument('-f', "--fragments", help="first n fragments to cut for classifying", default=5, required=False)
    parser.add_argument('-vt', "--volume_threshold", help="threshold to cut sound", default=10000, required=False)
    parser.add_argument('-sf', "--single_file_mode", help="process one file", default='off', required=False)
    parser.add_argument('-wp', "--wav_path", help="path to file in sf mode", default=None, required=False)
    parser.add_argument('-t', "--temp_path", help="path to dir for temp files", default='temp', required=False)
    parser.add_argument('-ct', "--clear_time", help="path to dir for temp files", default=3600, required=False)


    args = parser.parse_args()

    main(args)

    clean_temprorary(args.temp_path)

