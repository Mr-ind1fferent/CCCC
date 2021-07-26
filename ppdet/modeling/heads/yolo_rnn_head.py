import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


class ZDnet(nn.Layer):

    def __init__(self, input_size=4 * 200, hidden_size=40, num_layers=2):
        super(ZDnet, self).__init__()
        self.LSTM = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.Li = nn.Linear(hidden_size, input_size)
        self.sigoid = nn.Sigmoid()

    def forward(self, input):
        input = paddle.tensor.reshape(input, [1, 1, -1])
        x, ht = input[:, :, 0:800], input[:, :, 800:840]
        x, h = self.LSTM(x, ht)
        ht = h[-1][-1]
        x = paddle.tensor.reshape(ht, [1, -1])
        x = self.Li(x)
        x = self.sigoid(x)

        return x, ht  # x=[1, 800] ht=[1, 40]


@register
class YOLOv3Head(nn.Layer):
    __shared__ = ['num_classes', 'data_format']
    __inject__ = ['loss']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4,
                 data_format='NCHW',

                 ht=paddle.zeros([1, 1, 40])):
        """
        Head for YOLOv3 network

        Args:
            num_classes (int): number of foreground classes
            anchors (list): anchors
            anchor_masks (list): anchor masks
            loss (object): YOLOv3Loss instance
            iou_aware (bool): whether to use iou_aware
            iou_aware_factor (float): iou aware factor
            data_format (str): data format, NCHW or NHWC
        """
        super(YOLOv3Head, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels

        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)
        self.data_format = data_format

        self.yolo_outputs = []

        self.rnn = ZDnet()
        self.ht = ht
        self.con1 = nn.Conv1D(in_channels=1, out_channels=1, kernel_size=4, stride=4)
        # self.con2 = nn.Conv1D(in_channels=256,out_channels=1024,kernel_size=4,)
        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            conv = nn.Conv2D(
                in_channels=self.in_channels[i],
                out_channels=num_filters,
                kernel_size=1,
                stride=1,
                padding=0,
                data_format=data_format,
                bias_attr=ParamAttr(regularizer=L2Decay(0.)))
            conv.skip_quant = True
            yolo_output = self.add_sublayer(name, conv)
            self.yolo_outputs.append(yolo_output)

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None,
                bbox=paddle.normal(mean=paddle.full((1, 1, 800), 0.0), std=paddle.full((1, 1, 800), 0.01))):
        assert len(feats) == len(self.anchors)

        rnn_out, self.ht = self.rnn(paddle.concat([bbox, self.ht], axis=2))
        self.ht = self.ht.unsqueeze(0).unsqueeze(0)
        rnn_out = rnn_out[0]
        rnn_out.stop_gradient = True
        self.ht.stop_gradient = True

        yolo_outputs = []
        for i, feat in enumerate(feats):
            if (feat.shape[1] == 1024):

                rnn_out_1 = rnn_out.unsqueeze(0).unsqueeze(0)
                out_1024 = self.con1(rnn_out_1)
                out_1024 = F.sigmoid(out_1024)
                out_1024 = paddle.reshape(out_1024, [200, 1, 1])
                feat = feat + paddle.concat([out_1024, paddle.normal(
                    mean=paddle.full([feat.shape[1] - out_1024.shape[0], 1, 1], 0.0),
                    std=paddle.full([feat.shape[1] - out_1024.shape[0], 1, 1], 0.05))])

            else:
                rnn_out_512 = rnn_out.unsqueeze(0).unsqueeze(0)
                out_512 = self.con1(rnn_out_512)
                out_512 = F.sigmoid(out_512)
                out_512 = paddle.reshape(out_512, [200, 1, 1])
                feat = feat + paddle.concat([out_512, paddle.normal(
                    mean=paddle.full([feat.shape[1] - out_512.shape[0], 1, 1], 0.0),
                    std=paddle.full([feat.shape[1] - out_512.shape[0], 1, 1], 0.05))])

            yolo_output = self.yolo_outputs[i](feat)
            if self.data_format == 'NHWC':
                yolo_output = paddle.transpose(yolo_output, [0, 3, 1, 2])
            yolo_outputs.append(yolo_output)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors), yolo_outputs
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj ** (1 - self.iou_aware_factor)) * (
                            ioup ** self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }
