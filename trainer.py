import os
import glob
import time
import math
import numpy as np
from PIL import Image
from typing import Union, List
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
import torchsummary


class Sample:
    """
    This class represents either one sample or a batch of samples.
    A sample keeps both input and target tensors as well as auxiliary
    payload such as sample name.
    """

    def __init__(
            self,
            image: Union[np.ndarray, List[np.ndarray]],
            anno_hw: Union[np.ndarray, List[np.ndarray]],
            name: Union[str, List[str]],
            image_tensor: torch.Tensor,
            segmentation_tensor: torch.Tensor):
        """
        Constructor.
        :param image: original image
        :param anno_hw: annotated rectangle width/height
        :param name: text name of a sample
        :param image_tensor: input tensor to feed into NN
        :param segmentation_tensor: target for semantic segmentation
        """

        self.image = image
        self.anno_hw = anno_hw
        self.name = name
        self.image_tensor = image_tensor
        self.segmentation_tensor = segmentation_tensor

    def cuda(self):
        """ Push all tensors to cuda. """
        if self.image_tensor is not None:
            self.image_tensor = self.image_tensor.cuda()
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.cuda()

    def batchify(self):
        """ Create a batch out of one sample. """
        if self.image_tensor is not None:
            self.image_tensor = self.image_tensor.unsqueeze(dim=0)
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.unsqueeze(dim=0)

    @staticmethod
    def collate(batch_list):
        """ Create a batch out of a list of samples. """
        images = [t.image for t in batch_list]
        annos = [t.anno_hw for t in batch_list]
        image_tensor = torch.stack(
            [t.image_tensor for t in batch_list], dim=0)
        segmentation_tensor = torch.stack(
            [t.segmentation_tensor for t in batch_list], dim=0)
        names = [t.name for t in batch_list]
        batch_sample = Sample(
            images,
            annos,
            names,
            image_tensor,
            segmentation_tensor)
        return batch_sample


class CustomDataset:
    """ Class that reads samples of a custom dataset from a disk and preprocesses them. """

    def __init__(self, folder):
        self._ext = ".png"
        self._folder = folder
        name_list = glob.glob(os.path.join(folder, "*" + self._ext))
        raw_name_list = [os.path.splitext(os.path.split(p)[-1])[0] for p in name_list]
        self._name_list = [n for n in raw_name_list if self._parse_name(n) is not None]
        if len(self._name_list) == 0:
            print("Warning: custom dataset not found!")

    def get_list(self):
        return self._name_list

    def get_item(self, item):
        path = os.path.join(self._folder, item + self._ext)
        image_pil = Image.open(path)
        image_hwc = np.asarray(image_pil)
        image_hwc = image_hwc[:, :, :3] # throw away alpha channel
        anno_hw = self._parse_name(item)
        image_chw = np.transpose(image_hwc, (2, 0, 1)) / 255.0
        image_tensor_chw = torch.from_numpy(image_chw).type(torch.float32)
        return Sample(image_hwc, anno_hw, item, image_tensor_chw, None)

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
    """
    Since the custom dataset is too small, we train on a generated images with
    presumably wider representativeness of samples.
    """

    def __init__(self, image_shape_chw):
        self._image_shape_chw = image_shape_chw

    def generate(self):
        image_chw = np.zeros(self._image_shape_chw, dtype=np.float32)
        segmentation_tensor = np.zeros(self._image_shape_chw[1:], dtype=np.float32)
        anno_hw = np.zeros(4, dtype=np.int)

        h, w = self._image_shape_chw[1:]

        min_h, min_w = h // 5, w // 5

        color_bg = np.random.randint(256, size=3)
        color_fg = np.random.randint(256, size=3)
        color_bg = np.reshape(color_bg, (-1, 1, 1))
        color_fg = np.reshape(color_fg, (-1, 1, 1))
        # Continue mining until a big enough sample is found
        while True:
            left, right = np.sort(np.random.randint(w, size=2))
            top, bottom = np.sort(np.random.randint(h, size=2))
            if right - left < min_w:
                continue
            if bottom - top < min_h:
                continue
            break

        # Create input image
        image_chw[...] = color_bg
        image_chw[:, top:bottom, left:right] = color_fg

        # Create target segmentation
        segmentation_tensor[top:bottom, left:right] = 1.0

        anno_hw[0] = bottom - top
        anno_hw[1] = right - left

        image_float_chw = image_chw / 255
        image_hwc = np.transpose(image_chw, (1, 2, 0))
        image_tensor_chw = torch.from_numpy(image_float_chw)
        segmentation_tensor = torch.from_numpy(segmentation_tensor)

        return Sample(image_hwc, anno_hw, "synthetic", image_tensor_chw,
                      segmentation_tensor)


class DatasetDispatcher:
    """
    Class to multiplex synthetic and custom datasets.
    Provides generators for train and validation batches.
    """

    def __init__(
            self,
             work_shape = (3, 128, 128) # work resolution of a NN
             ):

        self._custom_dataset = CustomDataset("data/")
        self._image_shape = work_shape
        self._synthetic_dataset = SyntheticDataset(self._image_shape)
        self._val_names = self._custom_dataset.get_list()
        print("Custom dataset size =", len(self._val_names))

    def _tensor_custom_to_work_reso(self, tensor):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self._image_shape[1:],
            mode='nearest').squeeze(0)
        return tensor

    def tensor_work_to_custom(self, tensor):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self._custom_dataset.get_image_shape()[1:],
            mode='nearest').squeeze(0)
        return tensor

    def train_gen(self, batches_per_epoch, batch_size):
        """ Train generator returns full-size batches. """
        for i in range(batches_per_epoch):
            sample_list = []
            for ib in range(batch_size):
                sample = self._synthetic_dataset.generate()
                sample_list.append(sample)
            batch = Sample.collate(sample_list)
            yield batch

    def val_gen(self):
        """ Validation generator returns quasi-batches of size 1. """
        for name in self._val_names:
            sample = self._custom_dataset.get_item(name)
            sample.image_tensor = \
                self._tensor_custom_to_work_reso(sample.image_tensor)
            yield sample

    def get_image_shape(self):
        return self._image_shape


def Conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1)


def Conv3x3(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True)


def Upconv2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2)


class DownConv(nn.Module):
    """ Block UNet encoder. """

    def __init__(self, in_channels, out_channels, has_pool=True):
        super().__init__()

        self._has_pool = has_pool

        self._conv1 = Conv3x3(in_channels, out_channels)
        self._conv2 = Conv3x3(out_channels, out_channels)
        if self._has_pool:
            self._pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        shortcut = x
        if self._has_pool:
            x = self._pool(x)
        return x, shortcut


class UpConv(nn.Module):
    """ Block of UNet decoder. """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self._upconv = Upconv2x2(in_channels, out_channels)
        self._conv1 = Conv3x3(2 * out_channels, out_channels)
        self._conv2 = Conv3x3(out_channels, out_channels)

    def forward(self, shortcut, decoder_path):

        decoder_path = self._upconv(decoder_path)
        x = torch.cat((decoder_path, shortcut), 1)
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        return x


class UNet(nn.Module):
    """ Implementation of UNet based on https://arxiv.org/abs/1505.04597 """

    def __init__(
            self,
            out_channels,
            in_channels,
            num_levels=5,
            start_channels=64):

        super().__init__()

        self._out_channels = out_channels
        self._in_channels = in_channels
        self._start_channels = start_channels
        self._depth = num_levels

        down_convs = []
        up_convs = []

        # Create encoder blocks
        outs = None
        for i in range(num_levels):
            ins = self._in_channels if i == 0 else outs
            outs = self._start_channels * (2 ** i)
            pooling = True if i < num_levels - 1 else False

            down_conv = DownConv(ins, outs, has_pool=pooling)
            down_convs.append(down_conv)

        # Create decoder blocks
        for i in range(num_levels - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            up_convs.append(up_conv)

        self._last_conv = Conv1x1(outs, self._out_channels)

        self._down_convs = nn.ModuleList(down_convs)
        self._up_convs = nn.ModuleList(up_convs)

        self._init_weights()

    def _init_weights(self):
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_outs = []

        for _, module in enumerate(self._down_convs):
            x, shortcut = module(x)
            encoder_outs.append(shortcut)

        for i, module in enumerate(self._up_convs):
            shortcut = encoder_outs[-(i + 2)]
            x = module(shortcut, x)

        x = self._last_conv(x)

        return x


class Net(nn.Module):
    """ The entire NN. Encapsulates UNet and corresponding loss. """

    def __init__(self, input_shape_chw):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        start_channels = 8
        out_channels = 1

        num_levels = min(
            int(math.log2(self.input_shape_chw[1])),
            int(math.log2(self.input_shape_chw[2])),
        )
        print("num_levels = ", num_levels)

        self._net = UNet(
            out_channels, input_shape_chw[0],
            start_channels=start_channels, num_levels=num_levels)

        self._seg_loss = nn.modules.loss.BCELoss()

    def forward(self, image_tensor_batch: torch.Tensor):
        assert len(list(image_tensor_batch.size())) >= 3
        logits = self._net(image_tensor_batch)
        logits = logits.squeeze(1)
        pred = torch.sigmoid(logits)
        return pred

    def loss(self, pred: torch.Tensor, segmentation_gt: torch.Tensor):
        """
        Loss could be more complex, but we just go with
        vanilla binary cross entropy.
        """

        if segmentation_gt is not None:
            segmentation_loss = self._seg_loss(
                pred.view(-1),
                segmentation_gt.view(-1)
            )
        else:
            segmentation_loss = torch.zeros((1,), dtype=torch.float32)

        total_loss = segmentation_loss
        details = {
            "seg_loss": segmentation_loss.detach().cpu().item()
        }
        return total_loss, details


class Trainer:
    """ Class to manage train/validation cycles. """

    def __init__(self, load_last_snapshot=False):
        """
        Constructor.
        :param load_last_snapshot: whether to load the last snapshot from disk
        """

        has_gpu = torch.cuda.device_count() > 0
        if has_gpu:
            print(torch.cuda.get_device_name(0))
        else:
            print("GPU not found")
        self.use_gpu = has_gpu

        self._dispatcher = DatasetDispatcher()
        self._net = Net(self._dispatcher.get_image_shape())
        if self.use_gpu:
            self._net.cuda()

        self._snapshot_name = "snapshot.pth"
        if load_last_snapshot:
            load_kwargs = {} if self.use_gpu else {'map_location': 'cpu'}
            self._net.load_state_dict(torch.load(self._snapshot_name, **load_kwargs))

        # Print summary in Keras style
        shape_chw = tuple(self._dispatcher.get_image_shape())
        print("Work image shape =", shape_chw)
        torchsummary.summary(self._net, input_size=shape_chw)
        pass

    def train(self):
        """ Perform training of the network. """

        num_epochs = 50
        batch_size = 16
        batches_per_epoch = 1024
        learning_rate = 0.02

        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [40, 45], gamma=0.1, last_epoch=-1)

        training_start_time = time.time()

        self.validate()

        for epoch in range(num_epochs):
            print("Epoch ------ ", epoch)

            train_gen = self._dispatcher.train_gen(batches_per_epoch, batch_size)

            self._net.train()

            for batch_index, batch in enumerate(train_gen):
                if self.use_gpu:
                    batch.cuda()

                pred = self._net.forward(batch.image_tensor)

                loss, details = self._net.loss(pred, batch.segmentation_tensor)

                if batch_index % 50 == 0:
                    print("epoch={} batch={} loss={:.4f}".format(
                        epoch, batch_index, loss.item()
                    ))
                    self._render_prediction(
                        pred.detach().cpu().numpy()[0],
                        batch.segmentation_tensor.detach().cpu().numpy()[0],
                        batch.image_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0)))
                    print("-------------------------------")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

            scheduler.step()

            # Save after every epoch
            torch.save(self._net.state_dict(), self._snapshot_name)

            # Validate every epoch
            self.validate()

            pass
            # end of epoch

        training_end_time = time.time()
        print("Training took {} hours".format(
            (training_end_time - training_start_time)/3600))

        print("Train finished!")

    def validate(self):
        """ Validation cycle. Performed over a custom dataset. """
        print("Validation")

        self._net.eval()

        val_gen = self._dispatcher.val_gen()
        relative_error_list = []

        for sample_idx, sample in enumerate(val_gen):
            if self.use_gpu:
                sample.cuda()
            sample.batchify()

            pred = self._net.forward(sample.image_tensor)
            loss, details = self._net.loss(pred, sample.segmentation_tensor)

            def decode_prediction(pred: torch.Tensor) -> np.ndarray:
                pred_custom_res = self._dispatcher.tensor_work_to_custom(pred)
                mask = pred_custom_res > 0.5
                area = mask.sum()
                return area.detach().cpu().item()

            pred_area = decode_prediction(pred)
            anno_hw = sample.anno_hw
            gt_area = anno_hw[0] * anno_hw[1]
            relative_error = abs(pred_area - gt_area) / gt_area
            relative_error_list.append(relative_error)

            if sample_idx % 20 == 0:
                print("loss={:.4f} gt_area={} pred_area={}".format(
                    loss.item(), gt_area, pred_area
                ))
                self._render_prediction(
                    pred.detach().cpu().numpy()[0],
                    None,
                    sample.image_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0)))

        average_relative_error = \
            np.array(relative_error_list).sum() / len(relative_error_list)
        print("-------- Final metric -----------")
        print("Average relative area error = {:0.6f}".format(average_relative_error))

        pass

    def _render_prediction(self, pred: np.ndarray, gt: np.ndarray, input_image: np.ndarray):
        """
        This function visualizes predictions. Works nicely only
        in ipython notebook thus commented out here.
        """

        # % matplotlib inline
        # import matplotlib.pyplot as plt
        #
        # fig = plt.figure(figsize=(10, 3))
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(pred, cmap='gray', vmin=0.0, vmax=1.0)
        # fig.add_subplot(1, 3, 2)
        # if gt is not None:
        #     plt.imshow(gt, cmap='gray', vmin=0.0, vmax=1.0)
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(input_image, vmin=0.0, vmax=1.0)
        #
        # plt.show()

        pass


def main():
    """
    Default mode of operation is training. To run validation of a trained
    model over a custom dataset, run:
        trainer.py --validate
    """
    parser = ArgumentParser()
    parser.add_argument('--validate', default=False, action='store_true')
    args = parser.parse_args()

    if args.validate:
        trainer = Trainer(load_last_snapshot=True)
        trainer.validate()
    else:
        trainer = Trainer()
        trainer.train()
    pass


if __name__ == "__main__":
    main()

