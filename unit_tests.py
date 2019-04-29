import torch
import unittest
#from trainer import Decoder
import torch.nn as nn


class NetTest(unittest.TestCase):

    # def test_decoder(self):
    #     h, w = 16, 24
    #     decoder = Decoder()
    #     featuremap = torch.zeros(1, 4, h, w, dtype=torch.float32) - 3.0
    #     featuremap[:, Decoder.TL0, 2, 4] = 5.5
    #     featuremap[:, Decoder.TR1, 2, 20] = 6.6
    #     featuremap[:, Decoder.BL2, 10, 4] = 7.7
    #     featuremap[:, Decoder.BR3, 10, 20] = 8.8
    #     pred_hw = decoder.forward(featuremap)
    #     print(pred_hw)
    #     self.assertTrue(isinstance(pred_hw, torch.Tensor))
    #     self.assertAlmostEqual(pred_hw[0, 0].item(), 0.5000, places=3)
    #     self.assertAlmostEqual(pred_hw[0, 1].item(), 0.6666, places=3)
    #     return

    def test_bce(self):
        b, h, w = 160, 100, 200
        pred = torch.zeros(b, h, w, dtype=torch.float32)
        pred[:, :5, :6] = 1.0
        gt = torch.zeros(b, h, w, dtype=torch.long)
        loss_op = nn.BCELoss()
        loss = loss_op(pred, gt)
        print("Loss=", loss.item())
        pass


if __name__ == "__main__":
    unittest.main()
