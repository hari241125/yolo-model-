import torch
import torch.nn as nn
import numpy as np

class ConvBNLeaky(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        self.conv1 = ConvBNLeaky(3, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBNLeaky(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = ConvBNLeaky(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = ConvBNLeaky(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = ConvBNLeaky(128, 256, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = ConvBNLeaky(256, 512, 3, 1, 1)
        # self.pool6 = nn.MaxPool2d(2, 1, padding=1)  # remove

        self.conv7 = ConvBNLeaky(512, 1024, 3, 1, 1)

        # Detection head 1
        self.conv8 = ConvBNLeaky(1024, 256, 1, 1, 0)
        self.conv9 = ConvBNLeaky(256, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 255, 1, 1, 0)  # no BN, linear

        # Detection head 2
        self.route = self.conv8  # route from conv8
        self.conv11 = ConvBNLeaky(256, 128, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # route -1, 8 which is upsample and conv5 output
        self.conv12 = ConvBNLeaky(128 + 256, 256, 3, 1, 1)  # concat
        self.conv13 = nn.Conv2d(256, 255, 1, 1, 0)  # no BN

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.pool1(x1)

        x2 = self.conv2(x1)
        x2 = self.pool2(x2)

        x3 = self.conv3(x2)
        x3 = self.pool3(x3)

        x4 = self.conv4(x3)
        x4 = self.pool4(x4)

        x5_pre_pool = self.conv5(x4)
        x5 = self.pool5(x5_pre_pool)

        x6 = self.conv6(x5)
        # x6 = self.pool6(x6)

        x7 = self.conv7(x6)

        # Head 1
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        out1 = self.conv10(x9)

        # Head 2
        x11 = self.conv11(x8)
        x11_up = self.upsample(x11)
        x_cat = torch.cat([x11_up, x5_pre_pool], dim=1)
        x12 = self.conv12(x_cat)
        out2 = self.conv13(x12)

        return out1, out2

    def load_darknet_weights(self, weight_file):
        with open(weight_file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                conv = module
                if conv.bias is not None:
                    # Conv with bias
                    nb = conv.bias.numel()
                    conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+nb]))
                    ptr += nb
                # Conv weights
                nw = conv.weight.numel()
                conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+nw]).view(conv.weight.shape))
                ptr += nw
            elif isinstance(module, nn.BatchNorm2d):
                # BN: bias, weight, mean, var
                nb = module.bias.numel()
                module.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+nb])); ptr += nb
                module.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+nb])); ptr += nb
                module.running_mean.copy_(torch.from_numpy(weights[ptr:ptr+nb])); ptr += nb
                module.running_var.copy_(torch.from_numpy(weights[ptr:ptr+nb])); ptr += nb

if __name__ == "__main__":
    model = YOLOv3Tiny()
    model.load_darknet_weights('yolov3-tiny.weights')
    model.eval()

    # Save PyTorch model
    torch.save(model, 'yolov3-tiny.pt')

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 416, 416)
    torch.onnx.export(model, dummy_input, 'yolov3-tiny.onnx', opset_version=11)
    print("Exported to yolov3-tiny.onnx")