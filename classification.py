# -*- coding: utf-8 -*- 
import sys
import time
import os
import argparse

import neural_module

#sys.stdout.reconfigure(encoding='utf-8')

"""
DEFAULT CONFIGURATION:

    @IN_DIR (current as default) -
            путь к директории с входными изображениями
    @OUT_DIR (current + /output as default) -
            путь к директории с выходными JSON файлами
    @LOOP_MODE (False as default) - 
            True  - режим работы приложения на ожидание новых файлов в директории (бесконечный цикл)
            False - скрипт выполняется единожды для директории и не ожидает появления новых файлов
    @LOOP_TIMER (10 as default) -
            integer - количество секунд для повторного поиска новых файлов в режиме LOOP MODE
    @MODEL (required!) -
            путь к файлу обученной модели
    @KERAS_THRESHOLD (0.5 as default) -
            float от 0 до 1. Рекомендуется оставить 0.5
              
"""

def loadModel(weights):
    model = neural_module.createModel()
    trainedModel = False
    try:
        trainedModel = neural_module.loadWeights(model, weights)
    except Exception as e:
        raise('Ошибка при загрузке обученной модели:\n' + str(e))
    return trainedModel


def processSingleImage(image, trainedModel, threshold):
    result = False
    try:
        result = neural_module.classifySingleImage(image, trainedModel, threshold)
    except PermissionError:
        print('Image not ready now and will be processed in next iteration.')
        raise PermissionError
    except Exception as e:
        sys.exit('Ошибка при классификации изображения:\n' + str(e))
    return result



