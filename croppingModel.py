import torch
import torch.nn as nn
import torchvision.models as models
from roi_align.modules.roi_align import RoIAlignAvg, RoIAlign
from rod_align.modules.rod_align import RoDAlignAvg, RoDAlign
import torch.nn.init as init

class vgg_base(nn.Module):

    def __init__(self, loadweights=True, model_path=None):
        super(vgg_base, self).__init__()

        self.model_path = model_path

        vgg = models.vgg16()

        if loadweights:
            #print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

        self.VGG_base = nn.Sequential(*list(vgg.features._modules.values())[:-3])

    def forward(self, x):
        return self.VGG_base(x)



def dimRed_layer(reddim = 32):

    return nn.Conv2d(512, reddim, kernel_size=1, padding=0)


def fc_layers(reddim = 32, poolsize = 8):
    conv6 = nn.Conv2d(reddim, 1024, kernel_size=poolsize, padding=0)
    conv7 = nn.Conv2d(1024, 128, kernel_size=1)
    dropout = nn.Dropout(p=0.5)
    conv8 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv6, nn.ReLU(inplace=True),
               conv7, nn.ReLU(inplace=True),
               dropout, conv8)
    return layers


class crop_model(nn.Module):

    def __init__(self, poolsize = 8, reddim = 32, loadweight = True, model_path = None):
        super(crop_model, self).__init__()

        self.Feat_ext = vgg_base(loadweight, model_path)
        self.DimRed = dimRed_layer(reddim)
        self.RoIAlign = RoIAlignAvg(poolsize, poolsize, 1.0/16.0)
        self.RoDAlign = RoDAlignAvg(poolsize, poolsize, 1.0/16.0)
        self.FC_layers = fc_layers(reddim*2, poolsize)

    def forward(self, im_data, boxes):

        base_feat = self.Feat_ext(im_data)
        red_feat = self.DimRed(base_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        cat_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(cat_feat)
        return prediction


    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def build_crop_model(poolsize = 8, reddim = 32, loadweight = True, model_path=None):

    return crop_model(poolsize, reddim, loadweight, model_path)
