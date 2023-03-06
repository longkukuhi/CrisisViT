
from modeling.mae import models_vit
from torch import nn
import torch.nn.functional as F
import torch
from timm.models.layers import create_classifier
import timm

def prepare_mae_model(model_name='vit_base_patch16', chkpt_dir='mae_finetuned_vit_base.pth',):
    model = models_vit.__dict__[model_name](
        drop_path_rate=0.1,
        global_pool=True,
    )
    # load model
    if chkpt_dir is not None:
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        print("Load pre-trained checkpoint of MAE from: %s" % chkpt_dir)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print("message of loading MAE model: \n",msg)
        # if self.config['mae']['global_pool']:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    return model

def prepare_vit_model(image_res):

    if image_res == 224:
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        print("The resolution of image is 224")
    elif image_res == 384:
        model = timm.create_model('vit_base_patch16_384', pretrained=True)
        print("The resolution of image is 384")
    return model


class CrisisCNN(nn.Module):
    def __init__(self, model_name, config, task='incidence'):
        super(CrisisCNN, self).__init__()

        self.image_res = config['image_res']
        self.task = task

        self.img_encoder = self.prepare_img_encoder(model_name, config)

        self.sigmoid = nn.Sigmoid()

    def prepare_img_encoder(self, model_name, config):
        if model_name == 'resnet34':
            if self.task == 'incidence':
                self.img_encoder = timm.create_model('resnet34', pretrained=True,  num_classes=43)
                self.global_pool_inci, self.fc_inci = create_classifier(512, 43,
                                                                        pool_type='avg')

            elif self.task == 'place':
                self.img_encoder = timm.create_model('resnet34', pretrained=True, num_classes=49)
                self.global_pool_plac, self.fc_plac = create_classifier(512, 49,
                                                                        pool_type='avg')

            elif self.task == 'multi':
                self.img_encoder = timm.create_model('resnet34', pretrained=True, num_classes=1)

                self.global_pool_inci, self.fc_inci = create_classifier(512, 43,
                                                              pool_type='avg')
                self.global_pool_plac, self.fc_plac = create_classifier(512, 49,
                                                              pool_type='avg')
                # self.cls_head_inci = nn.Linear(512*49, 43)
                # self.cls_head_plac = nn.Linear(512*49, 49)

        elif model_name == 'resnet101d':
            if self.task == 'incidence':
                self.img_encoder = timm.create_model('resnet101d', pretrained=True, num_classes=43)
                self.global_pool_inci, self.fc_inci = create_classifier(512*4, 43,
                                                                        pool_type='avg')

            elif self.task == 'place':
                self.img_encoder = timm.create_model('resnet101d', pretrained=True, num_classes=49)
                self.global_pool_plac, self.fc_plac = create_classifier(512*4, 49,
                                                                        pool_type='avg')

            elif self.task == 'multi':
                self.img_encoder = timm.create_model('resnet101d', pretrained=True, num_classes=1)
                self.global_pool_inci, self.fc_inci = create_classifier(512*4, 43,
                                                                        pool_type='avg')
                self.global_pool_plac, self.fc_plac = create_classifier(512*4, 49,
                                                                        pool_type='avg')


        return self.img_encoder

    def forward(self, image, labels_inci=None, labels_plac=None, task='incidence', train=True):
        # image: (batch_size, 3, image_res, image_res)
        if task == 'multi':
            out = self.img_encoder.forward_features(image)
            out_inci = self.global_pool_inci(out)
            # out = out.flatten(1)
            out_inci = self.fc_inci(out_inci)

            out_plac = self.global_pool_plac(out)
            out_plac = self.fc_plac(out_plac)

            # out_inci = self.cls_head_inci(out)
            # out_plac = self.cls_head_plac(out)

            if train:
                loss_incidents = F.cross_entropy(out_inci, labels_inci)
                loss_places = F.cross_entropy(out_plac, labels_plac)
                return loss_incidents, loss_places
            else:
                return out_inci, out_plac

        else:
            if task == 'incidence':
                out = self.img_encoder.forward_features(image)
                # out_inci = self.cls_head_inci(out)
                out_inci = self.global_pool_inci(out)
                out_inci = self.fc_inci(out_inci)

                if train:
                    loss = F.cross_entropy(out_inci, labels_inci)
                    return loss
                else:
                    # out = self.sigmoid(out)
                    return out_inci

            elif task == 'place':
                out = self.img_encoder.forward_features(image)
                # out_plac = self.cls_head_plac(out)
                out_plac = self.global_pool_plac(out)
                out_plac = self.fc_plac(out_plac)
                if train:
                    loss = F.cross_entropy(out_plac, labels_plac)
                    return loss
                else:
                    return out_plac
            else:
                raise ValueError('task should be incidence or place.')


class CrisisVit(nn.Module):
    def __init__(self, model_name, config):
        super(CrisisVit, self).__init__()

        self.image_res = config['image_res']

        self.img_encoder = self.prepare_img_encoder(model_name, config)

        self.cls_head_inci = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], 43)
        )

        self.cls_head_plac = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['hidden_size'], 49)
        )

        self.sigmoid = nn.Sigmoid()

    def prepare_img_encoder(self, model_name, config):
        if model_name == 'ViT':
            self.img_encoder = prepare_vit_model(self.image_res)
        elif model_name == 'MAE':
            self.img_encoder = prepare_mae_model()
        return self.img_encoder

    def forward(self, image, labels_inci=None, labels_plac=None, task='incidence', train=True):
        # image: (batch_size, 3, image_res, image_res)
        if task == 'multi':
            out = self.img_encoder.forward_features(image)
            preds_inci = self.cls_head_inci(out)
            preds_plac = self.cls_head_plac(out)
            if train:
                loss_incidents = F.cross_entropy(preds_inci, labels_inci)
                loss_places = F.cross_entropy(preds_plac, labels_plac)
                return loss_incidents, loss_places
            else:
                return preds_inci, preds_plac

        elif task == 'incidence':
            out = self.img_encoder.forward_features(image)
            preds_inci = self.cls_head_inci(out)
            if train:
                loss_incidents = F.cross_entropy(preds_inci, labels_inci)
                return loss_incidents
            else:
                return preds_inci

        elif task == 'place':
            # out = self.mae_model.forward_features(image)
            out = self.img_encoder.forward_features(image)
            preds_plac = self.cls_head_plac(out)
            if train:
                loss_places = F.cross_entropy(preds_plac, labels_plac)
                return loss_places
            else:
                return preds_plac

        else:
            raise NotImplementedError

    def get_mae_model(self):
        return self.img_encoder

    def get_vit_model(self):
        return self.vit_model

class CrisisVitPretrainBinary(nn.Module):
    def __init__(self, model_name, config):
        super(CrisisVitPretrainBinary, self).__init__()

        self.config = config
        self.image_res = config['image_res']


        if model_name ==  'MAE':
            self.visual_encoder = self.prepare_mae_model('vit_base_patch16','mae_finetuned_vit_base.pth')
            print("We load the visual encoder weights from MAE")

        elif model_name == 'ViT':
            self.visual_encoder = self.prepare_vit_model(config)
            print("We load the visual encoder weights from ViT")

        self.cls_head = nn.Linear(config['hidden_size'], 2)
        self.cls_head.apply(self.init_weights)


    def forward(self, image, labels, train=True):
        # image: (batch_size, 3, image_res, image_res)
        if train:
            out = self.visual_encoder.forward_features(image)
            preds = self.cls_head(out)
            loss = F.cross_entropy(preds, labels)
            return loss
        else:
            out = self.visual_encoder.forward_features(image)
            preds = self.cls_head(out)
            return preds



    def prepare_mae_model(self, model_name, chkpt_dir, ):
        model = models_vit.__dict__[model_name](
            drop_path_rate=self.config['mae']['drop_path'],
            global_pool=self.config['mae']['global_pool'],
        )
        # load model
        if chkpt_dir is not None:
            checkpoint = torch.load(chkpt_dir, map_location='cpu')
            print("Load pre-trained checkpoint of MAE from: %s" % chkpt_dir)
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print("message of loading MAE model: \n", msg)

            # if self.config['mae']['global_pool']:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        return model

    def prepare_vit_model(self, config, visual_model_weight=None):

        if config['image_res'] == 224:
            model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
            print("The resolution of image is 224")
        elif config['image_res'] == 384:
            model = timm.create_model('vit_base_patch16_384', pretrained=True)
            print("The resolution of image is 384")
        return model

    def create_binary_cls_head(self):
        self.cls_head = nn.Linear(self.config['hidden_size'], 2).to('cuda')

        self.cls_head.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class CrisisVitDs(nn.Module):
    def __init__(self, model_name, config):
        super(CrisisVitDs, self).__init__()

        self.image_res = config['image_res']
        self.config = config

        self.cls_head_dt = nn.Linear(config['hidden_size'], 7)

        self.cls_head_info = nn.Linear(config['hidden_size'], 2)

        self.cls_human = nn.Linear(config['hidden_size'], 4)

        self.cls_damage = nn.Linear(config['hidden_size'], 3)

        self.apply(self.init_weights)

        if model_name == 'MAE':
            self.visual_encoder = self.prepare_mae_model('vit_base_patch16', 'mae_finetuned_vit_base.pth')
            print("We load the visual encoder weights from MAE")

        elif model_name == 'ViT':
            self.visual_encoder = self.prepare_vit_model(config)
            print("We load the visual encoder weights from ViT")

        elif model_name =='resnet':
            self.visual_encoder = self.prepare_resnet_model()
            print("We load the visual encoder weights from ResNet")

    def prepare_resnet_model(self):
        model = timm.create_model('resnet18', pretrained=True)
        return model



    def forward(self, image, labels=None, task='info', train=True):
        # image: (batch_size, 3, image_res, image_res)
        if task == 'info':
            out = self.visual_encoder.forward_features(image)
            preds_info = self.cls_head_info(out)
            if train:
                loss = F.cross_entropy(preds_info, labels)
                return loss
            else:
                return preds_info

        elif task == 'damage':
            out = self.visual_encoder.forward_features(image)
            preds_damage = self.cls_damage(out)
            if train:
                loss = F.cross_entropy(preds_damage, labels)
                return loss
            else:
                return preds_damage

        elif task == 'human':
            out = self.visual_encoder.forward_features(image)
            preds_human = self.cls_human(out)
            if train:
                loss = F.cross_entropy(preds_human, labels)
                return loss
            else:
                return preds_human

        elif task == 'dt':
            out = self.visual_encoder.forward_features(image)
            preds_dt = self.cls_head_dt(out)
            if train:
                loss = F.cross_entropy(preds_dt, labels)
                return loss
            else:
                return preds_dt

        else:
            raise NotImplementedError

    def prepare_mae_model(self, model_name, chkpt_dir, ):
        model = models_vit.__dict__[model_name](
            drop_path_rate=self.config['mae']['drop_path'],
            global_pool=self.config['mae']['global_pool'],
        )
        # load model
        if chkpt_dir is not None:
            checkpoint = torch.load(chkpt_dir, map_location='cpu')
            print("Load pre-trained checkpoint of MAE from: %s" % chkpt_dir)
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print("message of loading MAE model: \n", msg)

            # if self.config['mae']['global_pool']:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        return model

    def prepare_vit_model(self, config, visual_model_weight=None):

        if config['image_res'] == 224:
            model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
            print("The resolution of image is 224")
        elif config['image_res'] == 384:
            model = timm.create_model('vit_base_patch16_384', pretrained=True)
            print("The resolution of image is 384")
        return model

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()