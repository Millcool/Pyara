import unittest

import torch

from Model.model import ResNetBlock, MFCCModel, model_eval
from main import predict_audio


class TestResNetBlock(unittest.TestCase):
    def setUp(self):
        self.block = ResNetBlock(16, 16)

    def test_forward(self):
        batch_size = 16
        in_depth = 16
        signal = torch.randn(batch_size, in_depth, 4, 4)
        output = self.block(signal)
        self.assertEqual(output.shape, (batch_size, in_depth, 2, 2))


class TestMFCCModel(unittest.TestCase):
    def setUp(self):
        self.model = MFCCModel()

    def test_forward(self):
        batch_size = 8
        in_channels = 1
        T = 100
        signal = torch.randn(batch_size, in_channels, T)

        output = self.model(signal)

        self.assertEqual(output.shape, (batch_size, 2))


class TestModelEval(unittest.TestCase):
    def test_model_eval(self):
        model = model_eval()
        self.assertIsInstance(model, MFCCModel)


class TestAudioFormats(unittest.TestCase):
    def test_mp3(self):
        self.assertIn(predict_audio('test_audio/1.mp3'), (0, 1))

    def test_wav(self):
        self.assertIn(predict_audio('test_audio/1.wav'), (0, 1))

    def test_flac(self):
        self.assertIn(predict_audio('test_audio/1.flac'), (0, 1))

    def test_aiff(self):
        self.assertIn(predict_audio('test_audio/1.aiff'), (0, 1))

    def test_ogg(self):
        self.assertIn(predict_audio('test_audio/1.ogg'), (0, 1))

    def test_wma(self):
        self.assertIn(predict_audio('test_audio/1.wma'), (0, 1))

    def test_mp4(self):
        self.assertIn(predict_audio('test_audio/1.mp4'), (0, 1))

    def test_mp3_wav(self):
        self.assertIn(predict_audio('test_audio/2.wav'), (0, 1))


if __name__ == '__main__':
    unittest.main()