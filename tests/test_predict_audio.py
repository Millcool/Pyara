"""
Unit tests for audio prediction
"""
import unittest

from pyara.main import predict_audio


class TestPredictAudio(unittest.TestCase):
    def test_real_voice(self):
        file_path = "mozila11_1.wav"

        result = predict_audio(file_path)

        self.assertEqual(result, 0, "Expected real voice prediction")

    def test_synthesized_voice(self):
        file_path = "Alg_1_0.wav"

        result = predict_audio(file_path)

        self.assertEqual(result, 1, "Expected synthesized voice prediction")


if __name__ == '__main__':
    unittest.main()
