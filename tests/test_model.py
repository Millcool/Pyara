import torch
import unittest
from pyara.Model.model import ResNetBlock, MFCCModel, model_eval

class TestResNetBlock(unittest.TestCase):
    def setUp(self):
        self.block = ResNetBlock(16, 16)

    def test_forward(self):
        batch_size = 8
        in_depth = 16
        T = 10
        signal = torch.randn(batch_size, in_depth, T)

        output = self.block(signal)

        self.assertEqual(output.shape, (batch_size, 16, T/2))


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


if __name__ == '__main__':
    unittest.main()