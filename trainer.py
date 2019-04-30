import os
import sys
import glob
import math
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, List
import torchsummary


class Sample:
    def __init__(
            self,
            image: Union[np.ndarray, List[np.ndarray]],
            anno_hw: Union[np.ndarray, List[np.ndarray]],
            name: Union[str, List[str]],
            image_tensor: torch.Tensor,
            segmentation_tensor: torch.Tensor):

        self.image = image
        self.anno_hw = anno_hw
        self.name = name
        self.image_tensor = image_tensor
        self.segmentation_tensor = segmentation_tensor

    def cuda(self):
        if self.image_tensor is not None:
            self.image_tensor = self.image_tensor.cuda()
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.cuda()

    def batchify(self):
        if self.image_tensor is not None:
            self.image_tensor = self.image_tensor.unsqueeze(dim=0)
        if self.segmentation_tensor is not None:
            self.segmentation_tensor = self.segmentation_tensor.unsqueeze(dim=0)

    @staticmethod
    def collate(batch_list):
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
        while True:
            left, right = np.sort(np.random.randint(w, size=2))
            top, bottom = np.sort(np.random.randint(h, size=2))
            if right - left < min_w:
                continue
            if bottom - top < min_h:
                continue
            break

        image_chw[...] = color_bg
        image_chw[:, top:bottom, left:right] = color_fg

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
    def __init__(
            self,
             # work_shape = (3, 256, 256)
             work_shape = (3, 32, 32)  # converges!
             # work_shape = (3, 64, 64) # converges pretty fast
             #work_shape=(3, 128, 128)  # converges too slowly
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
        for i in range(batches_per_epoch):
            sample_list = []
            for ib in range(batch_size):
                sample = self._synthetic_dataset.generate()
                sample_list.append(sample)
            batch = Sample.collate(sample_list)
            yield batch

    def val_gen(self):
        for name in self._val_names:
            sample = self._custom_dataset.get_item(name)
            sample.image_tensor = self._tensor_custom_to_work_reso(sample.image_tensor)
            yield sample

    def get_image_shape(self):
        return self._image_shape


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     groups=groups,
                     stride=1)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels,
                                  out_channels,
                                  kernel_size=2,
                                  stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_mode='concat',
                 up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels,
                                self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels,
                                 self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet3rdParty(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super().__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class Net(nn.Module):
    def __init__(self, input_shape_chw):
        super().__init__()

        self.input_shape_chw = input_shape_chw

        featuremap_depth = 1  # 2 # 1  # 4

        num_levels = min(
            int(math.log2(self.input_shape_chw[1])),
            int(math.log2(self.input_shape_chw[2])),
        )
        print("num_levels=", num_levels)

        start_filts = 8  # 16

        # self._net = UNet(input_shape_chw, featuremap_depth)
        self._net = UNet3rdParty(featuremap_depth, input_shape_chw[0],
                                 start_filts=start_filts, depth=num_levels)

        # self._seg_loss = nn.modules.loss.MSELoss()
        self._seg_loss = nn.modules.loss.BCELoss()
        # self._seg_loss = nn.modules.loss.BCEWithLogitsLoss()

    def forward(self, image_tensor_batch: torch.Tensor):
        assert len(list(image_tensor_batch.size())) >= 3
        logits = self._net(image_tensor_batch)
        logits = logits.squeeze(1)
        pred = torch.sigmoid(logits)
        return pred

    def loss(self, pred: torch.Tensor, segmentation_gt: torch.Tensor):
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
    def __init__(self, load_last_snapshot=False):
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
            self._net.load_state_dict(torch.load(self._snapshot_name))

        shape_chw = tuple(self._dispatcher.get_image_shape())
        print("Work image shape =", shape_chw)
        torchsummary.summary(self._net, input_size=shape_chw)
        pass

    def train(self):
        num_epochs = 50  # 20 epochs for 32x32 model
        batch_size = 16
        batches_per_epoch = 1024
        learning_rate = 0.05

        optimizer = torch.optim.SGD(self._net.parameters(), lr=learning_rate)

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
                    print("epoch={} batch={} loss={:.4f} details={}".format(
                        epoch, batch_index,
                        loss.item(), details
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

            torch.save(self._net.state_dict(), self._snapshot_name)

            self.validate()
        pass

    def validate(self):
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
                print(sample.anno_hw)
                print("loss={:.4f} details={} gt_area={} pred_area={}".format(
                    loss.item(), details, gt_area, pred_area
                ))
                self._render_prediction(
                    pred.detach().cpu().numpy()[0],
                    None,
                    sample.image_tensor.detach().cpu().numpy()[0].transpose((1, 2, 0)))

        average_relative_error = \
            np.array(relative_error_list).sum() / len(relative_error_list)
        print("average_relative_error={:0.6f}".format(average_relative_error))

        pass

    def _render_prediction(self, pred: np.ndarray, gt: np.ndarray, input_image: np.ndarray):
        # % matplotlib inline
        # import matplotlib.pyplot as plt
        #
        # print("pred_mean={} gt_mean={}".format(
        #     pred.mean(), gt.mean() if gt is not None else float('nan')))
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
    if True:
        trainer = Trainer()
        trainer.train()
    else:
        trainer = Trainer(load_last_snapshot=True)
        trainer.validate()
    pass


if __name__ == "__main__":
    main()





