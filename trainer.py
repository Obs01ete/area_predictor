import os
import sys
import glob
import math
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

    def cuda(self):
        self.image_tensor = self.image_tensor.cuda()
        self.anno_hw_frac_tensor = self.anno_hw_frac_tensor.cuda()


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


class Split:
    def __init__(self, dataset: Dataset, val_frac: float):
        self._dataset = dataset
        all_names = self._dataset.get_list()
        num_val = int(val_frac * len(all_names))
        self._train_names = all_names[num_val:]
        print("Train size =", len(self._train_names))
        self._val_names = all_names[:num_val]
        print("Val size =", len(self._val_names))

    def train_gen(self, num_samples_in_epoch):
        for i in range(num_samples_in_epoch):
            index = i % len(self._train_names)
            name = self._train_names[index]
            yield self._dataset.get_item(name, do_augmentation=True)

    def val_gen(self):
        for name in self._val_names:
            yield self._dataset.get_item(name)


class Decoder:
    TL0 = 0
    TR1 = 1
    BL2 = 2
    BR3 = 3

    def __init__(self):
        pass

    @staticmethod
    def _rectify(v):
        return nn.ReLU()(torch.sigmoid(v) * 2 - 1)
        # return torch.sigmoid(v)

    def forward(self, featuremap):
        featuremap = self._rectify(featuremap)
        featuremap_left = featuremap[:, 0::2].sum(dim=1)
        featuremap_right = featuremap[:, 1::2].sum(dim=1)
        featuremap_top = featuremap[:, 0:2].sum(dim=1)
        featuremap_bottom = featuremap[:, 2:4].sum(dim=1)
        featuremap_w = torch.stack((featuremap_left, featuremap_right), dim=1)
        featuremap_h = torch.stack((featuremap_top, featuremap_bottom), dim=1)
        sum_w = torch.mean(featuremap_w, dim=2)
        sum_h = torch.mean(featuremap_h, dim=3)
        cumsum_w = sum_w.cumsum(dim=2)
        cumsum_h = sum_h.cumsum(dim=2)
        cumsum_norm_w = cumsum_w / (cumsum_w[:, :, -1].unsqueeze(dim=-1) + 1e-6)
        cumsum_norm_h = cumsum_h / (cumsum_h[:, :, -1].unsqueeze(dim=-1) + 1e-6)
        pred_profile_w = (cumsum_norm_w[:, 0, :] - cumsum_norm_w[:, 1, :]).squeeze(dim=1)
        pred_profile_h = (cumsum_norm_h[:, 0, :] - cumsum_norm_h[:, 1, :]).squeeze(dim=1)
        pred_w = pred_profile_w.mean(dim=1, keepdim=True)
        pred_h = pred_profile_h.mean(dim=1, keepdim=True)
        pred = torch.cat((pred_h, pred_w), dim=1)
        return pred


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


class SimpleBackbone(nn.Module):
    def __init__(self, input_shape_chw, featuremap_depth):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        ch = 16  # 32
        k = 3  # 3

        conv1 = Conv(input_shape_chw[0], ch, k)
        conv2 = Conv(ch, ch // 2, k)
        conv3 = Conv(ch // 2, featuremap_depth, k, has_relu=False)

        self._net = nn.Sequential(conv1, conv2, conv3)

    def forward(self, input):
        return self._net(input)


class UNet(nn.Module):
    def __init__(self, input_shape_chw, featuremap_depth):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        ch = 8
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
            #nn.Upsample(scale_factor=2, mode='bilinear')
            deconv = Deconv(ch, ch // 2, 3, stride=2)
            upscale_list.append(deconv)
            ch = ch // 2
        self._upscale_blocks = upscale_list

        self._out_conv = Conv(ch, featuremap_depth, 1)

        pass

    def forward(self, input):
        t = self._first_conv(input)
        branch_list = [t]
        for block in self._downscale_blocks:
            t = block(t)
            branch_list.append(t)

        branch_list = branch_list[:-1]

        for i_block, block in enumerate(self._upscale_blocks):
            upscaled_size = torch.Size((t.size()[2]*2, t.size()[3]*2))
            t = block(t, output_size=upscaled_size)
            t = t + branch_list[-i_block-1]
            pass

        t = self._out_conv(t)

        return t

class DecoderUNet():
    def __init__(self):
        pass

    def forward(self, featuremap):
        featuremap_sig = torch.sigmoid(featuremap)
        featuremap_w = featuremap_sig[:, 0, :, :].squeeze(1)
        featuremap_h = featuremap_sig[:, 1, :, :].squeeze(1)
        pred_w = featuremap_w.sum(dim=2).sum(dim=1, keepdim=True)
        pred_h = featuremap_h.sum(dim=2).sum(dim=1, keepdim=True)
        total_area = featuremap.size(2) * featuremap.size(3)
        pred_frac_w = pred_w / total_area
        pred_frac_h = pred_h / total_area
        pred = torch.cat((pred_frac_h, pred_frac_w), dim=1)
        return pred


class Net(nn.Module):
    def __init__(self, input_shape_chw):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        featuremap_depth = 2 # 1  # 4
        #self._net = SimpleBackbone(input_shape_chw, featuremap_depth)
        self._net = UNet(input_shape_chw, featuremap_depth)

        # maxpool_size = 8  # 4
        # self.maxpool = nn.MaxPool2d(maxpool_size, stride=maxpool_size)
        # featuremap_num_elems = \
        #     input_shape_chw[1] * input_shape_chw[2] * featuremap_depth // \
        #     (maxpool_size * maxpool_size)
        # self._head = nn.Linear(featuremap_num_elems, 2)

        self._loss = nn.MSELoss(reduction='sum')
        # self._loss = nn.L1Loss(reduction='sum')

    def forward(self, image_tensor_batch: torch.Tensor):
        assert len(image_tensor_batch.size()) >= 3
        featuremap = self._net(image_tensor_batch)

        # pred = Decoder().forward(featuremap)

        # featuremap_mp = self.maxpool(featuremap)
        # featuremap_flat = featuremap_mp.view(featuremap_mp.size(0), -1)
        # pred = self._head(featuremap_flat)

        pred = DecoderUNet().forward(featuremap)

        return pred

    def loss(self, pred: torch.Tensor, anno: torch.Tensor):
        assert len(pred.size()) == len(anno.size())
        assert pred.size()[-1] == 2
        assert anno.size()[-1] == 2
        loss_val = self._loss(pred, anno)
        return loss_val


class Trainer:
    def __init__(self):
        has_gpu = torch.cuda.device_count() > 0
        if has_gpu:
            print(torch.cuda.get_device_name(0))
        else:
            print("GPU not found")
        self.use_gpu = has_gpu
        dataset = Dataset("data/")
        self._split = Split(dataset, 0.1)
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
                image_tensor_batch = sample.image_tensor.unsqueeze(dim=0)
                anno_hw_frac_tensor_batch = sample.anno_hw_frac_tensor.unsqueeze(dim=0)

                pred = self._net.forward(image_tensor_batch)
                loss = self._net.loss(pred, anno_hw_frac_tensor_batch)
                if sample_idx % 250 == 0:
                    print("loss={:.4f} pred={} gt={}".format(
                        loss.item(), pred.detach().cpu().numpy(),
                        anno_hw_frac_tensor_batch.cpu().numpy()))
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
            image_tensor_batch = sample.image_tensor.unsqueeze(dim=0)
            anno_hw_frac_tensor_batch = sample.anno_hw_frac_tensor.unsqueeze(dim=0)

            pred = self._net.forward(image_tensor_batch)
            loss = self._net.loss(pred, anno_hw_frac_tensor_batch)
            if sample_idx % 3 == 0:
                print("loss={:.4f} pred={} gt={}".format(
                    loss.item(), pred.detach().cpu().numpy(),
                    anno_hw_frac_tensor_batch.cpu().numpy()))

        pass


def main():
    trainer = Trainer()
    trainer.train()
    pass


if __name__ == "__main__":
    main()


