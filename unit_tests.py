import unittest
from trainer import Predictor, CustomDataset


class NetTest(unittest.TestCase):

    def test_predictor(self):
        dataset = CustomDataset("data/")
        image = dataset.get_item(dataset.get_list()[0]).image

        predictor = Predictor()

        area = predictor(image)

        self.assertTrue(isinstance(area, int))
        self.assertGreater(area, 0)
        self.assertLessEqual(area, image.shape[0]*image.shape[1])

        print(area)
        pass


if __name__ == "__main__":
    unittest.main()
