"""Module for audio classification"""
from Model.model import model_eval
from audio_prepare import prediction, prepare_signal


def predict_audio(file_path):
    """
    Function for audio syntesized/real prediction

    :param file_path: path to the file
    :return: prediction about audio
    0: if real voice
    1: if syntesized voice
    """
    # Model to predict
    model = model_eval()
    signal = prepare_signal(file_path)

    pred = prediction(model, signal)
    return pred


predict_audio("mozila11_0.wav")
