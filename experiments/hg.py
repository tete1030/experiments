from __future__ import print_function, absolute_import
import torch
import pose.models as models
import pose.datasets as datasets
from pose.utils.transforms import fliplr_chwimg, fliplr_map
from pose.utils.evaluation import accuracy
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate

class Experiment:
    def __init__(self, hparams):
        model = models.HourglassNet(models.Bottleneck,
                                        num_stacks=hparams['model']['stack'],
                                        num_blocks=hparams['model']['block'],
                                        num_classes=datasets.mpii.POINT_NC)
        self.model = torch.nn.DataParallel(model).cuda()
    
        self.criterion = torch.nn.MSELoss(size_average=True).cuda()
        
        self.optimizer = torch.optim.RMSprop(list(model.parameters()),
                                             lr=hparams['learning_rate'],
                                             weight_decay=hparams['weight_decay'])

        self.train_dataset = datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                                           label_sigma=hparams['dataset']['label_sigma'],
                                           label_data=datasets.Mpii.LABEL_POINTS_MAP,
                                           label_type=hparams['dataset']['label_type'],
                                           single_person=True,
                                           selective=hparams['dataset']['selective'],
                                           train=True,
                                           contrast_factor=hparams['dataset']['contrast_factor'],
                                           brightness_factor=hparams['dataset']['brightness_factor'])
    
        self.val_dataset = datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                                         label_sigma=hparams['dataset']['label_sigma'],
                                         label_data=datasets.Mpii.LABEL_POINTS_MAP,
                                         label_type=hparams['dataset']['label_type'],
                                         train=False,
                                         single_person=True)

        self.hparams = hparams
    
    def epoch(self, epoch):
        self.hparams['learning_rate'] = adjust_learning_rate(self.optimizer, epoch, self.hparams['learning_rate'], self.hparams['schedule'], self.hparams['lr_gamma'])
        # decay sigma
        label_sigma_decay = self.hparams['dataset']['label_sigma_decay']
        if label_sigma_decay > 0:
            self.train_dataset.label_sigma *= label_sigma_decay
            self.val_dataset.label_sigma *= label_sigma_decay
    
    def process(self, batch, train):
        inputs, target, meta = batch
        volatile = not train
        input_var = torch.autograd.Variable(inputs.cuda(), volatile=volatile)
        target_var = torch.autograd.Variable(target['points'].cuda(async=True), volatile=volatile)
    
        output = self.model(input_var)
        score_map = output[-1].data.cpu()
    
        loss = self.criterion(output[0], target_var)
        for j in range(1, len(output)):
            loss += self.criterion(output[j], target_var)
    
        if not train and self.hparams['model']['flip']:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr_chwimg(inputs.numpy())).float().cuda(), 
                    volatile=True
                )
            flip_output_var = self.model(flip_input_var)
            flip_output = fliplr_map(flip_output_var[-1].data.cpu(), datasets.mpii.flipIndex)
            score_map += flip_output
            score_map /= 2.
    
        acc = accuracy(score_map, target['points'], datasets.mpii.evalIndex)
    
        return loss, acc[0], meta['index'], score_map


