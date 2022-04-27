from icevision import models


class Model():
    def __init__(self):
        super().__init__()
        self.model_type = models.torchvision.mask_rcnn
        self.backbone = model_type.backbones.resnet34_fpn()
        return self.model_type, self.backbone