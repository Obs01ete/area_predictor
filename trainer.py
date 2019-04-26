import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Union

class Sample:
    def __init__(
            self, image: np.ndarray, anno_hw: np.ndarray, name: str,
            image_tensor: torch.Tensor, anno_hw_frac_tensor: torch.Tensor):
        self.image = image
        self.anno_hw = anno_hw
        self.name = name
        self.image_tensor = image_tensor
        self.anno_hw_frac_tensor = anno_hw_frac_tensor


class Dataset:
    def __init__(self, folder):
        self._ext = ".png"
        self._folder = folder
        name_list = glob.glob(os.path.join(folder, "*"+self._ext))
        raw_name_list = [os.path.splitext(os.path.split(p)[-1])[0] for p in name_list]
        self._name_list = [n for n in raw_name_list if self._parse_name(n) is not None]

    def get_list(self):
        return self._name_list

    def __getitem__(self, item):
        path = os.path.join(self._folder, item+self._ext)
        image_pil = Image.open(path)
        image_hwc = np.asarray(image_pil)
        # print(image_hwc.shape)
        anno_hw = self._parse_name(item)
        image_chw = np.transpose(image_hwc, (2, 0, 1)) / 255.0
        image_tensor_chw = torch.from_numpy(image_chw).type(torch.float32)
        anno_hw_frac = anno_hw / np.array(image_chw.shape[1:])
        anno_hw_tensor = torch.from_numpy(anno_hw_frac).type(torch.float32)
        return Sample(image_hwc, anno_hw, item, image_tensor_chw, anno_hw_tensor)

    def _parse_name(self, name: str) -> Union[np.ndarray, None]:
        word_list = name.split("_")
        if len(word_list) == 5:
            width = int(word_list[2])
            height = int(word_list[4])
            return np.array((height, width), dtype=np.int)
        else:
            return None


class Split:
    def __init__(self, dataset: Dataset, val_frac: float):
        self._dataset = dataset
        all_names = self._dataset.get_list()
        num_val = int(val_frac * len(all_names))
        self._train_names = all_names[num_val:]

    def train_gen(self):
        for name in self._train_names:
            yield self._dataset[name]


class Net:
    def __init__(self):
        ch = 16
        conv1 = nn.Conv2d(4, ch, 3, padding=1)
        relu1 = nn.ReLU()
        conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        relu2 = nn.ReLU()
        conv3 = nn.Conv2d(ch, 4, 3, padding=1)
        self._net = nn.Sequential(conv1, relu1, conv2, relu2, conv3)

        self.loss = nn.MSELoss(reduce=True, reduction='sum')

    def forward(self, image_tensor_batch: torch.Tensor):
        assert len(image_tensor_batch.size()) >= 3
        h, w = image_tensor_batch.size()[-2:]
        featuremap = self._net(image_tensor_batch)
        featuremap_h = torch.sigmoid(featuremap[:, 0:2])
        featuremap_w = torch.sigmoid(featuremap[:, 2:4])
        sum_h = torch.mean(featuremap_h, dim=2)
        sum_w = torch.mean(featuremap_w, dim=3)
        cumsum_h = sum_h.cumsum(dim=2)
        cumsum_w = sum_w.cumsum(dim=2)
        cumsum_norm_h = cumsum_h / (cumsum_h[:, :, -1].unsqueeze(dim=-1))
        cumsum_norm_w = cumsum_w / (cumsum_w[:, :, -1].unsqueeze(dim=-1))
        pred_profile_h = (cumsum_norm_h[:, 0, :] - cumsum_norm_h[:, 1, :]).squeeze(dim=1)
        pred_profile_w = (cumsum_norm_w[:, 0, :] - cumsum_norm_w[:, 1, :]).squeeze(dim=1)
        pred_h = pred_profile_h.mean(dim=1, keepdim=True)
        pred_w = pred_profile_w.mean(dim=1, keepdim=True)
        pred = torch.cat((pred_h, pred_w), dim=1)
        return pred

    def loss(self, pred: torch.Tensor, anno: torch.Tensor):
        assert len(pred.size()) == len(anno.size())
        assert pred.size()[-1] == 2
        assert anno.size()[-1] == 2
        loss_val = self.loss(pred, anno)
        return loss_val

    def parameters(self):
        return self._net.parameters()


class Trainer:
    def __init__(self):
        dataset = Dataset("data/")
        self._split = Split(dataset, 0.1)
        self._net = Net()

    def train(self):
        num_epochs = 1000

        optimizer = torch.optim.SGD(self._net.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            print("Epoch --- ", epoch)
            train_gen = self._split.train_gen()
            for sample in train_gen:
                # print(sample.name, sample.image.shape, sample.anno_hw,
                #       sample.image_tensor.size(), sample.anno_hw_frac_tensor)
                image_tensor_batch = sample.image_tensor.unsqueeze(dim=0)
                anno_hw_frac_tensor_batch = sample.anno_hw_frac_tensor.unsqueeze(dim=0)
                pred = self._net.forward(image_tensor_batch)
                loss = self._net.loss(pred, anno_hw_frac_tensor_batch)
                print("loss=", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pass
        pass


def main():
    trainer = Trainer()
    trainer.train()
    pass

if __name__ == "__main__":
    main()
