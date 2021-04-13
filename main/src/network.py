import torch
from torch import nn
from torch.nn import functional as F


class MultiChannelTweetynet(nn.Module):
    def __init__(
        self,
        device,
        n_ffts,
        num_classes,
        input_shapes,
        conv1_kernel_sizes,
        num_conv1_filters,
        conv1_stride_lens,
        conv1_pad_same,
        pool1_kernel_sizes,
        pool1_stride_lens,
        pool1_pad_same,
        conv2_kernel_size,
        num_conv2_filters,
        conv2_stride_len,
        conv2_pad_same,
        pool2_kernel_size,
        pool2_stride_len,
        pool2_pad_same,
    ):
        super().__init__()

        self.device = device
        self.n_ffts = n_ffts
        self.num_classes = num_classes
        self.input_shapes = input_shapes
        self.conv1_pad_same = conv1_pad_same
        self.pool1_pad_same = pool1_pad_same
        self.conv2_pad_same = conv2_pad_same
        self.pool2_pad_same = pool2_pad_same

        self.cnn1_layers = self.build_cnn1_layers(
            num_conv1_filters,
            conv1_kernel_sizes,
            conv1_stride_lens,
            pool1_kernel_sizes,
            pool1_stride_lens,
        )

        conv2_in_channels = self.get_conv2_in_channels(num_conv1_filters)

        self.cnn2 = self.build_cnn2(
            conv2_in_channels,
            num_conv2_filters,
            conv2_kernel_size,
            conv2_stride_len,
            pool2_kernel_size,
            pool2_stride_len,
        )

        num_rnn_features, labelvec_len = self.test_net_forward()
        self.labelvec_len = labelvec_len
        self.rnn = nn.LSTM(
            input_size=num_rnn_features,
            hidden_size=num_rnn_features,
            num_layers=1,
            dropout=0,
            bidirectional=True,
        ).to(self.device)

        self.fc = nn.Linear(num_rnn_features * 2, num_classes).to(self.device)

    def build_cnn1_layers(
        self,
        out_channels_dict,
        conv_kernel_sizes_dict,
        conv_stride_lens_dict,
        pool_kernel_sizes_dict,
        pool_stride_lens_dict,
    ):
        cnn1_layers = nn.ModuleDict()
        for nfft in self.n_ffts:
            layer = [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=out_channels_dict[nfft],
                    kernel_size=conv_kernel_sizes_dict[nfft],
                    stride=conv_stride_lens_dict[nfft],
                ),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=pool_kernel_sizes_dict[nfft],
                    stride=pool_stride_lens_dict[nfft],
                ),
            ]
            if self.conv1_pad_same:
                layer.insert(
                    0,
                    PadSame(conv_kernel_sizes_dict[nfft], conv_stride_lens_dict[nfft],),
                )
            if self.pool1_pad_same:
                layer.insert(
                    3,
                    PadSame(pool_kernel_sizes_dict[nfft], pool_stride_lens_dict[nfft]),
                )
            cnn1_layers[str(nfft)] = nn.Sequential(*layer).to(self.device)
        return cnn1_layers

    def build_cnn2(
        self,
        in_channels,
        out_channels,
        conv_kernel_size,
        conv_stride_len,
        pool_kernel_size,
        pool_stride_len,
    ):
        layer = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride_len,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride_len),
        ]
        if self.conv2_pad_same:
            layer.insert(
                0, PadSame(conv_kernel_size, conv_stride_len),
            )
        if self.pool2_pad_same:
            layer.insert(
                3, PadSame(pool_kernel_size, pool_stride_len),
            )
        return nn.Sequential(*layer).to(self.device)

    def test_net_forward(self):
        tmp = {}
        for nfft, shape in self.input_shapes.items():
            tmp[nfft] = torch.empty(shape).to(self.device)
        with torch.no_grad():
            tmp_out = self.apply_cnn1_layers(tmp)
            out = self.cnn2(tmp_out)
        out_size = out.size()
        labelvec_len = out_size[3]
        channels_out, freqbins_out = out_size[1], out_size[2]
        num_rnn_features = channels_out * freqbins_out
        return num_rnn_features, labelvec_len

    def get_conv2_in_channels(self, num_conv1_filters):
        conv2_in_channels = 0
        for nfft, num_filters in num_conv1_filters.items():
            if nfft in self.n_ffts:
                conv2_in_channels += num_filters
        return conv2_in_channels

    def apply_cnn1_layers(self, x):
        channels = []
        for nfft, spec in x.items():
            layer = self.cnn1_layers[str(nfft)]
            out = layer(spec)
            out = torch.squeeze(out, 1)
            channels.append(out)

        if len(channels) > 1:
            return torch.cat(channels, 1)
        else:
            return channels[0]

    def forward(self, x):
        cnn1_out = self.apply_cnn1_layers(x)
        features = self.cnn2(cnn1_out)
        features = features.view(features.shape[0], self.rnn.input_size, -1)
        features = features.permute(2, 0, 1)
        rnn_output, (_, _) = self.rnn(features)
        rnn_output = rnn_output.permute(1, 0, 2)
        logits = self.fc(rnn_output)
        return logits.permute(0, 2, 1)


class PadSame(nn.Module):
    def __init__(self, filters, strides, dilations=None):
        super().__init__()
        self.filters = filters
        self.strides = strides
        self.dilations = dilations

    def compute_padding(self):
        paddings = []
        for dim in range(2):
            if self.dilations is not None:
                filter_size = (self.filters[dim] - 1) * self.dilations[dim] + 1
            else:
                filter_size = self.filters[dim]
            total_padding = filter_size - 1
            additional_padding = int(total_padding % 2 != 0)
            paddings.append((total_padding, additional_padding))
        return paddings

    def forward(self, x):
        (
            (f_total_pad, f_additional),
            (t_total_pad, t_additional),
        ) = self.compute_padding()
        f_half_pad = f_total_pad // 2
        t_half_pad = t_total_pad // 2
        x = F.pad(
            x,
            [
                t_half_pad,
                t_half_pad + t_additional,
                f_half_pad,
                f_half_pad + f_additional,
            ],
        )
        return x
