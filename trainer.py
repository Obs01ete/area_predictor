import os
import sys
import glob
import math
import torch
import torch.nn as nn
import torch.nn.modules.loss
import numpy as np
from PIL import Image
from typing import Union


class Sample:
    def __init__(
            self, image: np.ndarray, anno_hw: np.ndarray, name: str,
            image_tensor: torch.Tensor, anno_hw_frac_tensor: torch.Tensor,
            segmentation_tensor: torch.Tensor = None):

        self.image = image
        self.anno_hw = anno_hw
        self.name = name
        self.image_tensor = image_tensor
        self.anno_hw_frac_tensor = anno_hw_frac_tensor  # may be in HWYX format
        self.segmentation_tensor = segmentation_tensor

    def cuda(self):
        self.image_tensor = self.image_tensor.cuda()
        self.anno_hw_frac_tensor = self.anno_hw_frac_tensor.cuda()
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.cuda()

    def batchify(self):
        self.image_tensor = self.image_tensor.unsqueeze(dim=0)
        self.anno_hw_frac_tensor = self.anno_hw_frac_tensor.unsqueeze(dim=0)
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.unsqueeze(dim=0)


class Dataset:
    def __init__(self, folder):
        self._ext = ".png"
        self._folder = folder
        name_list = glob.glob(os.path.join(folder, "*" + self._ext))
        raw_name_list = [os.path.splitext(os.path.split(p)[-1])[0] for p in name_list]
        self._name_list = [n for n in raw_name_list if self._parse_name(n) is not None]
        assert len(self._name_list) > 0

    def get_list(self):
        return self._name_list

    def get_item(self, item, do_augmentation=False):
        path = os.path.join(self._folder, item + self._ext)
        image_pil = Image.open(path)
        image_hwc = np.asarray(image_pil)
        alpha = image_hwc[:, :, 3]
        red = image_hwc[:, :, 0]
        image_hwc = image_hwc[:, :, :3]  # throw away alpha channel
        if do_augmentation:
            rand_h, rand_w = np.random.randint(2, size=2)
            if rand_h > 0:
                image_hwc = np.flip(image_hwc, axis=0)
            if rand_w > 0:
                image_hwc = np.flip(image_hwc, axis=1)
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

    def get_image_shape(self):
        sample = self.get_item(self._name_list[0])
        return list(sample.image_tensor.size())


class SyntheticDataset:
    def __init__(self, image_shape_chw):
        self._image_shape_chw = image_shape_chw

    def generate(self):
        image_tensor_chw = np.zeros(self._image_shape_chw, dtype=np.float32)
        segmentation_tensor = np.zeros(self._image_shape_chw[1:], dtype=np.float32)
        anno_hwyx_tensor = np.zeros(4, dtype=np.float32)

        h, w = self._image_shape_chw[1:]

        # TODO fill
        color_bg = np.random.randint(256, size=3)
        color_fg = np.random.randint(256, size=3)
        color_bg = np.reshape(color_bg, (-1, 1, 1))
        color_fg = np.reshape(color_fg, (-1, 1, 1))
        left, right = np.sort(np.random.randint(w, size=2))
        top, bottom = np.sort(np.random.randint(h, size=2))

        image_tensor_chw[...] = color_bg
        image_tensor_chw[:, top:bottom, left:right] = color_fg
        image_tensor_chw = image_tensor_chw / 255

        segmentation_tensor[top:bottom, left:right] = 1.0

        anno_hwyx_tensor[0] = (bottom - top) / h
        anno_hwyx_tensor[1] = (right - left) / w
        anno_hwyx_tensor[2] = (bottom + top) / (2 * h)
        anno_hwyx_tensor[3] = (right + left) / (2 * w)

        #         image_hwc = np.transpose(image_tensor_chw, (1, 2, 0))
        #         plt.figure()
        #         plt.imshow(image_hwc)
        #         plt.figure()
        #         plt.imshow(segmentation_tensor, cmap='gray')
        #         assert False

        image_tensor_chw = torch.from_numpy(image_tensor_chw)
        segmentation_tensor = torch.from_numpy(segmentation_tensor)
        anno_hwyx_tensor = torch.from_numpy(anno_hwyx_tensor)

        return Sample(None, None, "generated", image_tensor_chw, anno_hwyx_tensor,
                      segmentation_tensor=segmentation_tensor)


class Split:
    def __init__(self, dataset: Dataset, val_frac: float, synthetic_train: bool = False):
        self._dataset = dataset
        self._synthetic_dataset = SyntheticDataset(dataset.get_image_shape()) \
            if synthetic_train else None
        all_names = self._dataset.get_list()
        num_val = int(val_frac * len(all_names))
        self._train_names = all_names[num_val:]
        print("Train size =", len(self._train_names))
        self._val_names = all_names[:num_val]
        print("Val size =", len(self._val_names))

    def train_gen(self, num_samples_in_epoch):
        for i in range(num_samples_in_epoch):
            if self._synthetic_dataset is not None:
                yield self._synthetic_dataset.generate()
            else:
                index = i % len(self._train_names)
                name = self._train_names[index]
                yield self._dataset.get_item(name, do_augmentation=True)

    def val_gen(self):
        for name in self._val_names:
            yield self._dataset.get_item(name)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, has_relu=True):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) // 2, stride=stride)
        self._relu = nn.ReLU() if has_relu else None

    def forward(self, input):
        conv = self._conv(input)
        relu = self._relu(conv) if self._relu is not None else conv
        return relu


class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, has_relu=True):
        super().__init__()
        self._conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) // 2, stride=stride)
        self._relu = nn.ReLU() if has_relu else None

    def forward(self, input, **kwargs):
        conv = self._conv(input, **kwargs)
        relu = self._relu(conv) if self._relu is not None else conv
        return relu


class UNet(nn.Module):
    def __init__(self, input_shape_chw, featuremap_depth):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        ch = 4  # 8
        self._first_conv = Conv(input_shape_chw[0], ch, 3)
        branch_channels = [ch]

        num_levels = min(8, int(math.log2(min(input_shape_chw[1], input_shape_chw[2]))))
        downscale_list = nn.ModuleList()
        for _ in range(num_levels):
            conv1 = Conv(ch, ch, 3)
            conv2 = Conv(ch, ch * 2, 3, stride=2)
            downscale_list.append(nn.Sequential(conv1, conv2))
            branch_channels.append(ch)
            ch = ch * 2
        self._downscale_blocks = downscale_list

        upscale_list = nn.ModuleList()
        for _ in range(num_levels):
            deconv = Deconv(ch, ch // 2, 3, stride=2)
            upscale_list.append(deconv)
            ch = ch // 2
        self._upscale_blocks = upscale_list

        self._out_conv = Conv(ch, featuremap_depth, 1, has_relu=False)

        pass

    def forward(self, input):
        t = self._first_conv(input)
        branch_list = [t]
        for block in self._downscale_blocks:
            t = block(t)
            branch_list.append(t)

        branch_list = branch_list[:-1]

        for i_block, block in enumerate(self._upscale_blocks):
            upscaled_size = torch.Size((t.size()[2] * 2, t.size()[3] * 2))
            t = block(t, output_size=upscaled_size)
            t = t + branch_list[-i_block - 1]
            pass

        t = self._out_conv(t)

        return t


class Prediction:
    def __init__(self, regression: torch.Tensor, segmentation: torch.Tensor):
        self.regression = regression
        self.segmentation = segmentation


class FullConnectedRegressor(nn.Module):
    def __init__(self, featuremap_num_elems):
        super().__init__()

        inner = 64
        self._linear1 = nn.Linear(featuremap_num_elems, inner)
        self._relu1 = nn.ReLU()
        self._linear2 = nn.Linear(inner, 4)

    def forward(self, t):
        t = t.view(t.size(0), -1)
        t = self._linear1(t)
        t = self._relu1(t)
        t = self._linear2(t)
        return t


class Net(nn.Module):
    def __init__(self, input_shape_chw):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        featuremap_depth = 1  # 2 # 1  # 4
        self._net = UNet(input_shape_chw, featuremap_depth)

        featuremap_num_elems = \
            input_shape_chw[1] * input_shape_chw[2] * featuremap_depth
        self._fc_regressor = FullConnectedRegressor(featuremap_num_elems)

        self._reg_loss = nn.MSELoss(reduction='sum')

        # self._seg_loss = nn.modules.loss.BCEWithLogitsLoss()
        self._seg_loss = nn.modules.loss.MSELoss()

    def forward(self, image_tensor_batch: torch.Tensor):
        assert len(list(image_tensor_batch.size())) >= 3
        featuremap_logits = self._net(image_tensor_batch)
        featuremap = torch.sigmoid(featuremap_logits)
        regression = self._fc_regressor(featuremap)
        pred = Prediction(regression, featuremap)
        return pred

    def loss(self, pred: Prediction, anno: torch.Tensor, segmentation_gt: torch.Tensor):
        assert len(pred.regression.size()) >= len(anno.size())
        assert anno.size()[-1] >= 2
        pred_reg = pred.regression[:, :anno.size()[-1]]
        reg_loss = self._reg_loss(pred_reg, anno)
        if segmentation_gt is not None:
            segmentation_loss = self._seg_loss(pred.segmentation, segmentation_gt.unsqueeze(0))
        else:
            segmentation_loss = torch.zeros((1), dtype=torch.float32)
        # total_loss = reg_loss + segmentation_loss
        total_loss = segmentation_loss
        details = {
            # "reg_loss": reg_loss.detach().cpu().item(),
            "seg_loss": segmentation_loss.detach().cpu().item()
        }
        return total_loss, details


class Trainer:
    def __init__(self):
        has_gpu = torch.cuda.device_count() > 0
        if has_gpu:
            print(torch.cuda.get_device_name(0))
        else:
            print("GPU not found")
        self.use_gpu = has_gpu
        dataset = Dataset("data/")
        self._split = Split(dataset, 0.1, synthetic_train=True)
        self._net = Net(dataset.get_image_shape())
        if self.use_gpu:
            self._net.cuda()

    def train(self):
        num_epochs = 10000
        num_samples_in_epoch = 1000

        optimizer = torch.optim.SGD(self._net.parameters(), lr=0.01)

        self.validate()

        for epoch in range(num_epochs):
            print("Epoch --- ", epoch)
            train_gen = self._split.train_gen(num_samples_in_epoch)
            for sample_idx, sample in enumerate(train_gen):
                # print(sample.name, sample.image.shape, sample.anno_hw,
                #       sample.image_tensor.size(), sample.anno_hw_frac_tensor)

                if self.use_gpu:
                    sample.cuda()
                sample.batchify()

                pred = self._net.forward(sample.image_tensor)
                loss, details = self._net.loss(
                    pred, sample.anno_hw_frac_tensor, sample.segmentation_tensor)
                if sample_idx % 250 == 0:
                    print("loss={:.4f} details={} pred={} gt={}".format(
                        loss.item(), details, pred.regression.detach().cpu().numpy(),
                        sample.anno_hw_frac_tensor.cpu().numpy()))
                    self._render_prediction(
                        pred.segmentation.detach().cpu().numpy().squeeze(0).squeeze(0),
                        sample.segmentation_tensor.detach().cpu().numpy().squeeze(0),
                        sample.image_tensor.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0)))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

            self.validate()
        pass

    def validate(self):
        print("Validation")
        val_gen = self._split.val_gen()
        for sample_idx, sample in enumerate(val_gen):
            if self.use_gpu:
                sample.cuda()
            sample.batchify()

            pred = self._net.forward(sample.image_tensor)
            loss, details = self._net.loss(pred, sample.anno_hw_frac_tensor, sample.segmentation_tensor)
            if sample_idx % 3 == 0:
                print("loss={:.4f} details={} pred={} gt={}".format(
                    loss.item(), details,
                    pred.regression.detach().cpu().numpy(),
                    sample.anno_hw_frac_tensor.cpu().numpy()))

        pass

    def _render_prediction(self, pred: np.ndarray, gt: np.ndarray, input_image: np.ndarray):
        # % matplotlib inline
        # import matplotlib.pyplot as plt
        #
        # # print(pred.shape)
        # # print(gt.shape)
        #
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(pred, cmap='gray')
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(gt, cmap='gray')
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(input_image, vmin=0.0, vmax=1.0)
        #
        # plt.show()

        pass


def main():
    trainer = Trainer()
    trainer.train()
    pass


if __name__ == "__main__":
    main()



