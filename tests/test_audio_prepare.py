"""
Tests for audio prepare functions
"""

import torch
import torchaudio
import unittest
import numpy as np
from config import CFG
from pyara import cut_if_necessary, right_pad_if_necessary, prepare_signal,  prediction
from audio_prepare import MFCC_spectrogram, prediction_multiple, prepare_signals
from pyara.Model.model import model_eval

class TestFunctions(unittest.TestCase):
    def test_cut_if_necessary(self):
        # Test case when signal.shape[2] > CFG.width
        signal = torch.randn(2, 2, 400)
        expected_output = signal[:, :, 0:CFG.width]
        self.assertTrue(np.array_equal(cut_if_necessary(signal).numpy(), expected_output.numpy()))

        # Test case when signal.shape[2] <= CFG.width
        signal = torch.randn(2, 2, 50)
        self.assertTrue(np.array_equal(cut_if_necessary(signal).numpy(), signal.numpy()))

    def test_right_pad_if_necessary(self):
        """Тест функции дополняющей сигнал до нужной ширины, если ширина сигнала
         меньше чем CFG.width или другая заданная пользователем ширина сигнали"""
        # Тест когда length_signal < CFG.width
        signal = torch.randn(1, 1, 50)
        expected_output = torch.nn.functional.pad(signal, (0, CFG.width - 50))
        self.assertTrue(np.array_equal(right_pad_if_necessary(signal).numpy(), expected_output.numpy()))

        # Тест когда length_signal >= CFG.width
        signal = torch.randn(1, 1, 300)
        self.assertTrue(np.array_equal(right_pad_if_necessary(signal).numpy(), signal.numpy()))

    def test_prepare_signal(self):
        """Тест полного цикла предобработки аудио"""

        voice_path = "mozila11_1.wav"
        signal, sample_rate = torchaudio.load(voice_path)
        expected_output = signal.mean(dim=0)
        expected_output = expected_output.unsqueeze(dim=0)
        expected_output = MFCC_spectrogram(expected_output)
        expected_output = cut_if_necessary(expected_output)
        expected_output = right_pad_if_necessary(expected_output)
        expected_output = expected_output.repeat(3, 1, 1)
        expected_output = expected_output.unsqueeze(dim=0)
        expected_output = expected_output.to(CFG.device)

        self.assertTrue(np.array_equal(prepare_signal(voice_path).numpy(), expected_output.numpy()))

    def test_prediction_multiple(self):
        # Create a model for testing (you should replace this with your actual model)
        model = model_eval()
        # Можно доработать передавая больше параметров в функцию
        # Например громкость, MFCC/LFCC, Ширина окна, длина дополнения/обрезки
        signal = prepare_signals(['mozila11_1.wav', 'mozila11_1.wav'], 0, 300, 16000)

        prediction_of_model, probability = prediction_multiple(model, signal)
        print(prediction_of_model)
        self.assertEqual(prediction_of_model[0], 0)
        self.assertEqual(prediction_of_model[1], 0)

    def test_prediction(self):
        # Create a model for testing (you should replace this with your actual model)
        model = model_eval()

        signal = prepare_signal('mozila11_1.wav', 0, 300, 16000)

        prediction_of_model, probability = prediction(model, signal)
        print(prediction_of_model)
        self.assertEqual(prediction_of_model, 0)


if __name__ == '__main__':
    unittest.main()
