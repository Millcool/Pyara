import torch
import torchaudio
import unittest
from config import CFG
from audio_prepare import cut_if_necessary, right_pad_if_necessary, prepare_signal, prediction, MFCC_spectrogram
from Model.model import model_eval

class TestFunctions(unittest.TestCase):
    def test_cut_if_necessary(self):
        # Test case when signal.shape[2] > CFG.width
        signal = torch.randn(1, 1, 100)
        expected_output = signal[:, :, 0:CFG.width]
        self.assertEqual(cut_if_necessary(signal), expected_output)

        # Test case when signal.shape[2] <= CFG.width
        signal = torch.randn(1, 1, 50)
        self.assertEqual(cut_if_necessary(signal), signal)

    def test_right_pad_if_necessary(self):
        # Test case when length_signal < CFG.width
        signal = torch.randn(1, 1, 50)
        expected_output = torch.nn.functional.pad(signal, (0, CFG.width - 50))
        self.assertEqual(right_pad_if_necessary(signal), expected_output)

        # Test case when length_signal >= CFG.width
        signal = torch.randn(1, 1, 100)
        self.assertEqual(right_pad_if_necessary(signal), signal)

    def test_prepare_signal(self):
        voice_path = "../mozila11_0.wav"
        signal, sample_rate = torchaudio.load(voice_path)
        expected_output = signal.mean(dim=0)
        expected_output = expected_output.unsqueeze(dim=0)
        expected_output = MFCC_spectrogram(expected_output)
        expected_output = cut_if_necessary(expected_output)
        expected_output = right_pad_if_necessary(expected_output)
        expected_output = expected_output.repeat(3, 1, 1)
        expected_output = expected_output.unsqueeze(dim=0)
        expected_output = expected_output.to(CFG.device)

        self.assertEqual(prepare_signal(voice_path), expected_output)

    def test_prediction(self):
        model = model_eval()
        signal = torch.randn(1, 1, CFG.width)

        # Test case when output is 1
        model_output = torch.Tensor([[0.3, 0.7]])
        expected_output = 1
        with torch.no_grad():
            model_output = model_output.to(CFG.device)
            self.assertEqual(prediction(model, signal), expected_output)

        # Test case when output is 0
        model_output = torch.Tensor([[0.8, 0.2]])
        expected_output = 0
        with torch.no_grad():
            model_output = model_output.to(CFG.device)
            self.assertEqual(prediction(model, signal), expected_output)

if __name__ == '__main__':
    unittest.main()
