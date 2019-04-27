import torch
import unittest
from trainer import Decoder


class NetTest(unittest.TestCase):

    def test_decoder(self):
        h, w = 16, 24
        decoder = Decoder()
        featuremap = torch.zeros(1, 4, h, w, dtype=torch.float32) - 3.0
        featuremap[:, Decoder.TL0, 2, 4] = 5.5
        featuremap[:, Decoder.TR1, 2, 20] = 6.6
        featuremap[:, Decoder.BL2, 10, 4] = 7.7
        featuremap[:, Decoder.BR3, 10, 20] = 8.8
        pred_hw = decoder.forward(featuremap)
        print(pred_hw)
        self.assertTrue(isinstance(pred_hw, torch.Tensor))
        self.assertAlmostEqual(pred_hw[0, 0].item(), 0.5000, places=3)
        self.assertAlmostEqual(pred_hw[0, 1].item(), 0.6666, places=3)
        return


if __name__ == "__main__":
    unittest.main()
