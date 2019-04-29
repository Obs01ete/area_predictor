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
        if self.image_tensor is not None:
            self.image_tensor = self.image_tensor.cuda()
        if self.anno_hw_frac_tensor is not None:
            self.anno_hw_frac_tensor = self.anno_hw_frac_tensor.cuda()
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.cuda()

    def batchify(self):
        if self.image_tensor is not None:
            self.image_tensor = self.image_tensor.unsqueeze(dim=0)
        if self.anno_hw_frac_tensor is not None:
            self.anno_hw_frac_tensor = self.anno_hw_frac_tensor.unsqueeze(dim=0)
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.unsqueeze(dim=0)

    @staticmethod
    def collate(batch_list):
        image_tensor = torch.stack([t.image_tensor for t in batch_list], dim=0)
        segmentation_tensor = torch.stack([t.segmentation_tensor for t in batch_list], dim=0)
        batch_sample = Sample(
            None,
            None,
            "i_am_batchman",
            image_tensor,
            None,
            segmentation_tensor=segmentation_tensor)
        return batch_sample


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


class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self._upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self._upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self._upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._conv = Conv(in_channels, out_channels, 1)
        pass

    def forward(self, input, **kwargs):
        t = self._upsample(input)
        t = self._conv(t)
        return t


class UNet(nn.Module):
    def __init__(self, input_shape_chw, featuremap_depth):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        ch = 16  # 8
        self._first_conv = Conv(input_shape_chw[0], ch, 3)
        branch_channels = [ch]

        max_levels = 4  # 6
        num_levels = min(max_levels,
                         int(math.log2(min(input_shape_chw[1], input_shape_chw[2]))))
        downscale_list = nn.ModuleList()
        for _ in range(num_levels):
            conv1 = Conv(ch, ch, 3)
            conv2 = Conv(ch, ch * 2, 3, stride=2)
            downscale_list.append(nn.Sequential(conv1, conv2))
            branch_channels.append(ch)
            ch = ch * 2
        self._downscale_blocks = downscale_list

        print("branch_channels=", branch_channels)

        upscale_list = nn.ModuleList()
        fusion_list = nn.ModuleList()
        for _ in range(num_levels):
            # deconv = Deconv(ch, ch // 2, 3, stride=2)
            deconv = Upscale(ch, ch // 2)
            upscale_list.append(deconv)
            fusion = Conv(ch, ch // 2, 1)
            fusion_list.append(fusion)
            ch = ch // 2
        self._upscale_blocks = upscale_list
        self._fusion_blocks = fusion_list

        self._out_conv = Conv(ch, featuremap_depth, 1, has_relu=False)

        pass

    def forward(self, input):
        t = self._first_conv(input)
        # print("first_conv=", t.size())
        branch_list = [t]
        for block in self._downscale_blocks:
            t = block(t)
            # print("downscale_out=", t.size())
            branch_list.append(t)

        branch_list = branch_list[:-1]
        # for branch in branch_list:
        #    print("branch=", branch.size())

        for i_block, (upscale, fusion) in enumerate(zip(self._upscale_blocks, self._fusion_blocks)):
            upscaled_size = torch.Size((t.size()[2] * 2, t.size()[3] * 2))
            t = upscale(t, output_size=upscaled_size)
            # print("upscale_out=", t.size())
            shortcut = branch_list[-i_block - 1]
            # print("shortcut=", shortcut.size())
            concat = torch.cat((t, shortcut), dim=1)
            # concat = torch.cat((t, t), dim=1) ### DIRTY HACK
            # print("concat=", concat.size())
            t = fusion(concat)
            # print("fusion=", t.size())
            pass

        t = self._out_conv(t)

        # print("out_conv=", t.size())

        # assert False

        return t


class Prediction:
    def __init__(self, regression: torch.Tensor, segmentation: torch.Tensor):
        self.regression = regression
        self.segmentation = segmentation


class Net(nn.Module):
    def __init__(self, input_shape_chw):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        featuremap_depth = 1  # 2 # 1  # 4
        self._net = UNet(input_shape_chw, featuremap_depth)

        # self._seg_loss = nn.modules.loss.MSELoss()
        self._seg_loss = nn.modules.loss.BCELoss()
        # self._seg_loss = nn.modules.loss.BCEWithLogitsLoss()

    def forward(self, image_tensor_batch: torch.Tensor):
        assert len(list(image_tensor_batch.size())) >= 3
        featuremap_logits = self._net(image_tensor_batch)
        featuremap_logits = featuremap_logits.squeeze(1)
        featuremap = torch.sigmoid(featuremap_logits)
        # regression = self._fc_regressor(featuremap_logits)
        pred = Prediction(None, featuremap)
        return pred

    def decode(self, prediction: Prediction):
        # logits = prediction.segmentation
        # decoded = torch.sigmoid(logits)
        decoded = prediction.segmentation
        return decoded

    def loss(self, pred: Prediction, segmentation_gt: torch.Tensor):
        if segmentation_gt is not None:
            segmentation_loss = self._seg_loss(
                # pred.segmentation,
                # segmentation_gt
                pred.segmentation.view(-1),
                segmentation_gt.view(-1)
            )
        else:
            segmentation_loss = torch.zeros((1), dtype=torch.float32)

        total_loss = segmentation_loss
        details = {
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

        # import torchsummary
        # shape_chw = tuple(dataset.get_image_shape())
        # print("shape_chw=", shape_chw)
        # torchsummary.summary(self._net, input_size=shape_chw)
        pass

    def train(self):
        num_epochs = 10000
        num_samples_in_epoch = 1024 * 16

        batch_size = 32

        optimizer = torch.optim.SGD(self._net.parameters(), lr=0.01)

        self.validate()

        #         def batch_collator(sample_gen, batch_size):
        #             batch_list = []
        #             for sample in sample_gen:
        #                 batch_list.append(sample)
        #                 if len(batch_list) == batch_size:
        #                     batch = Sample.collate(batch_list)
        #                     yield batch

        for epoch in range(num_epochs):
            print("Epoch --- ", epoch)

            train_gen = self._split.train_gen(num_samples_in_epoch)

            # train_collator_gen = batch_collator(train_gen, batch_size)

            batch_list = []
            batch_index = 0
            for sample_idx, sample in enumerate(train_gen):

                if len(batch_list) < batch_size:
                    batch_list.append(sample)
                    # print("append")
                    continue

                # print("process ----")
                batch = Sample.collate(batch_list)
                batch_list = []
                batch_index = batch_index + 1

                # print(batch.image_tensor.size())

                if self.use_gpu:
                    batch.cuda()
                # sample.batchify()

                pred = self._net.forward(batch.image_tensor)
                decoded_segmentation = self._net.decode(pred)
                loss, details = self._net.loss(
                    pred, batch.segmentation_tensor)
                if batch_index % 10 == 0:
                    print("loss={:.4f} details={}".format(  # pred={} gt={}
                        loss.item(), details
                        # pred.regression.detach().cpu().numpy(),
                        # batch.anno_hw_frac_tensor.cpu().numpy()
                    ))
                    self._render_prediction(
                        decoded_segmentation.detach().cpu().numpy()[0],
                        batch.segmentation_tensor.detach().cpu().numpy()[0],
                        batch.image_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0)))
                    print("-------------------------------")
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
            loss, details = self._net.loss(pred, sample.segmentation_tensor)
            if sample_idx % 3 == 0:
                print("loss={:.4f} details={}".format(  # pred={} gt={}
                    loss.item(), details
                    # pred.regression.detach().cpu().numpy(),
                    # sample.anno_hw_frac_tensor.cpu().numpy()
                ))

        pass

    def _render_prediction(self, pred: np.ndarray, gt: np.ndarray, input_image: np.ndarray):
        # % matplotlib inline
        # import matplotlib.pyplot as plt
        #
        # print("pred_mean={} gt_mean={}".format(pred.mean(), gt.mean()))
        #
        # fig = plt.figure(figsize=(10, 3))
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(pred, cmap='gray', vmin=0.0, vmax=1.0)
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(gt, cmap='gray', vmin=0.0, vmax=1.0)
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




