"""
Module for audio classification
"""
from pyara.Model.model import model_eval
from pyara.audio_prepare import prediction, prepare_signal
from pyara.config import CFG


def predict_audio(file_path
                  ,print_probability = False
                  ,pitch_shift = 0
                  ,width = CFG.width
                  ,sample_rate = CFG.SAMPLE_RATE):
    """
     Функция для предсказания аудио (синтезированного / подлинного).

     Параметры:
         file_path (str): Путь к файлу.

     Возвращает:
         int: Предсказание аудио:
             0: если аудио подлинное
             1: если аудио синтезированное
     """

    # Model to predict
    model = model_eval()
    # Можно доработать передавая больше параметров в функцию
    # Например громкость, MFCC/LFCC, Ширина окна, длина дополнения/обрезки
    signal = prepare_signal(file_path,  pitch_shift, width, sample_rate)

    prediction_of_model, probability = prediction(model, signal)
    if print_probability:
        return f'Answer:{prediction_of_model} probability:{probability}'
    #Если print_probability = False
    return prediction_of_model


if __name__ == '__main__':
    print(predict_audio("mozila11_1.wav", print_probability=True, pitch_shift=10))
