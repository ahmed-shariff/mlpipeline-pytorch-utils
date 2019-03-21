import numpy as np
import math
import itertools
import os
import time
import statistics
import logging

from tqdm import tqdm

from mlpipeline.utils import add_script_dir_to_PATH
from mlpipeline.utils import ExecutionModeKeys
from mlpipeline.utils import Versions
from mlpipeline.helper import Experiment
from mlpipeline.helper import DataLoader
from mlpipeline.utils import version_parameters
from mlpipeline.utils import log
from mlpipeline.utils import console_colors
from mlpipeline.utils import ModeKeys

try:
    import cv2
    cv2_available = True
except ImportError:
    cv2_available = False
    log("opencv cannot be imported. Check for installation.", level = logging.WARN)
    
try:
    import torch
    import torch.nn as nn
    torch_available = True
except ImportError:
    torch_available = False
    log("pytorch cannot be imported. Check for installation.", level = logging.WARN)

    
class BaseTorchExperiment(Experiment):
    def __init__(self, versions, **args):
        super().__init__(versions, **args)
        self.model = None
        self.topk_k = None
        self.logging_iteration = None
        self.criterion = None
        self.optimizer = None
        self.checkpoint_saving_per_epoc = None
        self.use_cuda = None
        self.save_history_checkpoints_count = None
        

    def pre_execution_hook(self, version, experiment_dir, exec_mode=ExecutionModeKeys.TEST):
        #print("Pre execution")
        print("Version spec: ", version)
        self.current_version = version
        self.dataloader = self.current_version[version_parameters.DATALOADER]
        self.dataloader.set_classes()
        self.history_file_name = "{}/model_params{}.tch".format(experiment_dir.rstrip("/"), "{}")
        self.file_name = self.history_file_name.format(0)
        if os.path.isfile(self.file_name):
            self.log("Loading parameters from: {}".format(self.file_name))
            self.load_history_checkpoint(self.file_name)
            
        else:
            self.epocs_params = 0
            self.log("No checkpoint")
            
    def get_current_version(self):
        return self.current_version

    def get_trained_step_count(self):
        #TODO: +1 to epocs_params or not?
        ret_val =  (self.epocs_params) * self.dataloader.get_train_sample_count()/ \
            self.dataloader.batch_size
        self.log("steps_trained: {}".format(ret_val))
        return ret_val

    def train_loop(self, input_fn, steps):
        #print("steps: ", steps)
        #print("calling input fn")
        #criterion = torch.nn.CrossEntropyLoss()
        top1 = matricContainer()
        topk = matricContainer()
        loss_average = matricContainer()
        #if use_cuda:
        #    criterion.cuda()
        # optimizer = torch.optim.Adam(self.model.parameters(), 0.00001, (0.9,0.9))#
        #torch.optim.SGD(self.model.parameters(), 0.1)
        #print([f for f in self.model.parameters()])
        #torch.optim.Adam(self.model.parameters(), 0.1, (0.7,0.7))
        
        
        datasize = self.dataloader.get_train_sample_count()/self.dataloader.batch_size
        
        epocs = int(steps/datasize)#self.current_version[version_parameters.EPOC_COUNT]
        #print(datasize)
        if epocs == 0:
            epocs = 1
        self.log("Epocs: {}".format(epocs))
        self.model.train()
        for epoc in range(epocs):
            for idx, i in enumerate(input_fn):
                #print(idx)
                if idx > 1:
                    #break
                    pass
                if self.use_cuda:
                    input_var = torch.autograd.Variable(i[0].cuda())
                    label_var = torch.autograd.Variable(i[1].cuda())
                else:
                    input_var = torch.autograd.Variable(i[0])
                    label_var = torch.autograd.Variable(i[1])
                    pass
                out = self.model(input_var)
                loss = self.criterion(out, label_var)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print(next(self.model.fc.parameters()).data)
                #print(next(self.model.fc.parameters()).grad)
                
                t1, tk, total = accuracy(out, label_var, self.topk_k)
                top1.update(t1, total)
                topk.update(tk, total)
                loss_average.update(loss, 1)
            
                #print()rint(idx % self.logging_iteration)
                if idx % self.logging_iteration == 0:
                    #print(label_var)
                    out_string_step = "Step: {}   Epoc: {}".format((self.epocs_params +
                                                                    epoc) * datasize + idx,
                                                                   self.epocs_params + epoc + 1)
                    out_string_results="Top1: {:.4f}  Top{}: {:.4f}  Loss: {:.4f}  Loss_: {}".format(
                        top1.avg(),
                        self.topk_k,
                        topk.avg(),
                        loss_average.avg().data[0], 0)#loss.data)
                    self.log(out_string_step)
                    self.log(out_string_results)

            if epoc % self.checkpoint_saving_per_epoc == 0:
                self.save_checkpoint(epoc)
        #print(self.model.state_dict())
        return "{}  {}".format(out_string_results, out_string_step)
                
    def evaluate_loop(self, input_fn, steps):
        top1 = matricContainer()
        topk = matricContainer()
        loss_average = matricContainer()
        #print(self.model.state_dict())
        self.model.eval()
        for idx, i in tqdm(enumerate(input_fn)):
            #print(idx)
            if idx > 5:
                #break
                pass
            if self.use_cuda:
                input_var = torch.autograd.Variable(i[0].cuda())
                label_var = torch.autograd.Variable(i[1].cuda())
            else:
                input_var = torch.autograd.Variable(i[0])
                label_var = torch.autograd.Variable(i[1])
                pass
            out = self.model(input_var)
            #print("*"*20)
            #loss = self.criterion(out, label_var)
            t1, tk, total = accuracy(out, label_var, self.topk_k)
            top1.update(t1, total)
            self.model.zero_grad()
            topk.update(tk, total)
            #loss_average.update(loss, 1)
            #print()rint(idx % self.logging_iteration)
        out_string_step = "Step: {}".format(idx)
        out_string_results = "Top1: {:.4f}  Top{}: {:.4f}".format(
            top1.avg(),
            self.topk_k,
            topk.avg())
        self.log(out_string_step)
        self.log(out_string_results)
            #loss_average.avg().data[0]))
        return "{}  {}".format(out_string_results, out_string_step)

    def save_checkpoint(self, epoc):
        directory = os.path.dirname(self.file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
                
        if self.save_history_checkpoints_count is not None:
            if self.save_history_checkpoints_count < 1:
                raise ValueError("save_history_checkpoints_count should be 1 or higher. Else set it to None to completely disable this feature.")
            for history_idx in range(self.save_history_checkpoints_count - 1, -1, -1):
                history_file_name = self.history_file_name.format(history_idx)
                if os.path.exists(history_file_name):
                    os.replace(history_file_name, self.history_file_name.format(history_idx + 1))
            self.log("History checkpoints: {}".format(self.save_history_checkpoints_count))
        torch.save({
            'epoch': epoc,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'validation': self.dataloader.valid_data_filtered,
            'lr_scheduler': None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        }, self.file_name)
        self.log("Saved checkpoint for epoc: {} at {}".format(epoc + 1, self.file_name))
                    
    def load_history_checkpoint(self, checkpoint_file_name, load_optimizer = True):
        self.log("Loading: {}".format(checkpoint_file_name), log_to_file = True)
        checkpoint = torch.load(checkpoint_file_name)
        self.epocs_params = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['lr_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if checkpoint['validation'] is not None:
            self.dataloader.set_validation_set(checkpoint['validation'])

    def get_ancient_checkpoint_file_name(self, epoc_from_last = None):
        if epoc_from_last is None:
            epoc_from_last = self.save_history_checkpoints_count
        elif epoc_from_last > self.save_history_checkpoints_count:
            raise ValueError("`epoc_from_last` should be less than or equal `self.save_history_checkpoints_count`.")
        elif epoc_from_last == 0:
            history_file_name = self.history_file_name.format(0)
            if os.path.exists(history_file_name):
                return history_file_name
            
        if self.save_history_checkpoints_count < 1:
            raise ValueError("save_history_checkpoints_count should be 1 or higher. Else set it to None to completely disable this feature.")
        for history_idx in range(epoc_from_last, 0, -1):
            history_file_name = self.history_file_name.format(history_idx)
            if os.path.exists(history_file_name):
                return history_file_name


class Dataloader_multidataset(DataLoader):
    def __init__(self,
                 dataloaders):

        # for dl in dataloaders:
        #     if isinstance(dl, DataLoader):
        #         raise TypeError("All elements should be DataLoader")
        self.dataLoaders = dataloaders
        self.DATA_CODE_MAPPING = {idx: dl.DATA_CODE_MAPPING for idx, dl in enumerate(dataloaders)}
        self.summery = str({idx: dl.summery for idx, dl in enumerate(dataloaders)})
    def set_classes(self):
        self.batch_size = 0
        self.valid_data_filtered = []
        for dl in self.dataLoaders:
            dl.set_classes()
            self.batch_size += dl.batch_size
            self.valid_data_filtered.append(dl.valid_data_filtered)
        
    def get_train_input(self, mode= ModeKeys.TRAIN, **kargs):
        return [dl.get_train_input(mode, **kargs) for dl in self.dataLoaders]
        
    def get_test_input(self, **kargs):
        return [dl.get_test_input(**kargs) for dl in self.dataLoaders]

    def get_validation_input(self, **kargs):
        return [dl.get_validation_input(**kargs) for dl in self.dataLoaders]
    
    def get_dataloader_summery(self, **kargs):
        '''
        This function will be called to log a summery of the dataloader when logging the results of a model
        '''
        return self.summery

    def get_train_sample_count(self):
        '''
        returns the number of datapoints being used as the training dataset. This will be used to assess the number of epocs during training and evaluating.
        '''
        return sum([dl.get_train_sample_count() for dl in self.dataLoaders])

    def get_test_sample_count(self):
        '''
        returns the number of datapoints being used as the testing dataset. This will be used to assess the number of epocs during training and evaluating.
        '''
        return sum([dl.get_test_sample_count() for dl in self.dataLoaders])

    def set_validation_set(self,validation_data):
        for idx, data in enumerate(validation_data):
            self.dataLoaders[idx].set_validation_set(data)
    
    def log(self, message,log_to_file=False, **kargs):
        log("{}DataLoader- {}{}".format(console_colors.CYAN_FG,
                                   console_colors.RESET,
                                   message),
            log=log_to_file, **kargs)


# taken from the pytorch source https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResNet_feature_extraction(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_feature_extraction, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.level_layers = {
            0: [self.conv1, self.bn1, self.relu, self.maxpool],
            1: [self.layer1],
            2: [self.layer2],
            3: [self.layer3],
            4: [self.layer4]
        }
                
    def _make_layer(self, block, planes, blocks, stride=1, factorize = False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x, label = None, train_round = None, level = None, level_in = True):
        if level is None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x_avg = self.avgpool(x)
            return x_avg, x
        else:
            if level_in:
                r = range(level)
            else:
                r = range(level, len(self.level_layers))
            for i in r:
                for j in range(len(self.level_layers[i])):
                    x = self.level_layers[i][j](x)
            return x
        
    
def accuracy(output, labels, topk):
    _, prediction = output.topk(topk)
    top1 = []
    topk = []
    top1 = prediction[:,0].eq(labels)
    topk = torch.sum((prediction.t().eq(labels)), 0)
    
    try:
        t1 = top1.nonzero().size(0)
    except RuntimeError as e:
        print(e)
        t1 = 0
    try:
        tk = topk.nonzero().size(0)
    except RuntimeError as e:
        print(e)
        tk = 0
    return t1, tk, labels.size(0)


def accuracy_logits(output, labels, topk):
    _, prediction = output.topk(topk)
    if labels.dim() == 2:
        labels = labels.nonzero()[:,1]
    top1 = []
    topk = []
    top1 = prediction[:,0].eq(labels)
    # print(labels.size(), prediction.t().size())
    topk = torch.sum((prediction.t().eq(labels)), 0)
    #print("*"*20,prediction, labels)
    #print("@"*20, top1, topk)
    #print("#"*20, top1.nonzero().size(), topk.nonzero().size(), labels.size())
    try:
        t1 = top1.nonzero().size(0)
    except RuntimeError as e:
        print(e)
        t1 = 0
    try:
        tk = topk.nonzero().size(0)
    except RuntimeError as e:
        print(e)
        tk = 0
    return t1, tk, labels.size(0)


def accuracy_multiclass_hamming_loss(output, target, threshold = 0.75):
    output = output > threshold
    hamming_count = torch.sum(output != target.byte())
    hamming_total = target.size(0) * target.size(1)
    return hamming_count, hamming_total

def accuracy_multiclass(output, target, threshold = 0.75):
    output = output > threshold
    temp = output+target.byte()
    intersection = torch.sum(temp == 2, 1)
    #union = torch.sum(temp > 0, 1)
    length = torch.sum(target, 1)
    non_zero_targets = length != 0 
    return torch.sum(intersection[non_zero_targets].float()/length[non_zero_targets].float()), target.size(0)
    

def accuracy_class_counts(output, labels, selecting_target=None):
    output_round = output.round()
    t1 = 0
    
    if selecting_target is None:
        dense_idx = labels != 0
        result = output_round[dense_idx].eq(labels[dense_idx])
        t1 = result.sum().item()
        tot = dense_idx.sum().item()
    #TODO: correct this also to be used with multilabels settings    
    else:
        dense_idx = selecting_target
        for i, idx in enumerate(dense_idx):
            if output_round[i][idx] == labels[i]:
                t1 += 1
        tot = labels.size(0)
    return t1, tot

def accuracy_counts(output, labels):
    top1 = []
    top1 = output[:,0].round().eq(labels)
    
    try:
        t1 = top1.nonzero().size(0)
    except RuntimeError as e:
        print(e)
        t1 = 0
    return t1, labels.size(0)

def export_model(model,
                 data_codes_mapping,
                 used_labels,
                 export_path,
                 model_summery,
                 dataloader_summery):
    export_path_file = "{}/{}/model_params.tch".format(export_path.rstrip("/"), time.time())
    directory = os.path.dirname(export_path_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    log("Exporting model to : {}".format(export_path_file), modifier_1 = console_colors.GREEN_FG)
    log("-- Data codes mapping: {}".format(data_codes_mapping))
    log("-- Used labels: {}".format(used_labels))
    log("-- Model Summery: {}".format(model_summery))
    log("-- Dataloder Summery: {}".format(dataloader_summery))
    torch.save({
        # 'model': model
        'state_dict': model.state_dict(),
        'data_codes_mapping': data_codes_mapping,
        'used_labels': used_labels,
        'model_summery': model_summery,
        'dataloader_summery': dataloader_summery
        },
        export_path_file)

def crop_to_box(img, size):
    shape = img.shape
    if shape[0] > shape[1]:
        start = int((shape[0]-shape[1])/2)
        img = img[start:start+shape[1], :]
    else:
        start = int((shape[1]-shape[0])/2)
        img = img[:, start:start+shape[0]]
    
    img = cv2.resize(img, (size, size))
    return img

class log_with_temp_file():
    def __init__(self, file_path, log_fn):
        self.file_path = file_path
        self.log_fn = log_fn
        open(self.file_path, "w")
        
    def __call__(self, message, *args, **kwargs):
        log(message, *args, **kwargs)
        with open(self.file_path, "a") as f:
            f.write(str(message) + "\n")


def _eval_model(log_fn, classes, used_labels,
                json_in, predict_on_model, root_dir,
                total_classes_count, prob_threshold = 0.1,
                calc_negative_class = False, multi_image = False,
                show_wrongs = False):
    #classes, used_labels, predict_on_model =
    #predict_on_model(multi_image = multi_image)
    classes_reversed = {v:k for k,v in classes.items()}
    classes = {c:classes[c] for c in [classes_reversed[c_] for c_ in used_labels]}
    exact_match = 0
    hemming = 0
    accuracy = 0

    # These dictionaries are used to keep track of the values which will
    # be used to calculate the metrics we need
    micro_matrics = {k:[0,0,0,0] for k in used_labels}# tp, fp, tn, fn
    macro_matrics = {k:[[0,0], # recall, 
                        [0,0], # prece, 
                        0, # acc, 
                        [0,0],# acc_non_tn, 
                        [0,0]] # [count eql, total]
                     for k in used_labels}# recall, prece, acc, acc_non_tn, [count eql, total]
    if calc_negative_class:
        micro_matrics['None'] = [0,0,0,0]
        macro_matrics['None'] = [[0,0],[0,0],0, [0,0], [0,0]]
    used_labels_length = len(used_labels)
    count_total_average = 0

    # Setting up the combintaion that need to be tracked.
    # Generating the combinations and initializing the respective metrics
    combinations_performance = list(itertools.combinations(used_labels, 2))
    combinations_performance = {k:[0,0] for k in combinations_performance}
    calced = 0
    i_ = 0
    for idx,row in tqdm(enumerate(json_in)):
        # if idx> 100:
        #     break
        if idx % 5000 == 0:
            # Added to give the hardware some breathing time to cool down
            time.sleep(3 * idx/10000)
        f = row[0]
        gt_labels = row[1]
        # This is when we are using images which do not have a label
        # Example: detecting negative classes
        if len(gt_labels) == 0:
            gt_labels = {'None':0}

        # Testing if we are encountering any instances with no label
        # if encountered, and calc_negative_class not true, will raise exception?
        try:
            for l in gt_labels.keys():
                if int(l) not in classes:
                    #log_fn("row")
                    raise Exception()
        except:
            if calc_negative_class:
                # comment this out if considering images that are not in used labels
                # continue
                pass
            else:
                # Should this be continue of raise?
                raise
            
        # classificatoin = []
        # count = []
        
        # The difference being if we are looking at singel image per data point or a scene(mulitiple images)
        if isinstance(f, str):
            img = cv2.imread(os.path.join(root_dir, f))
            x_classify, x_count, x_count_total, x_classify_aux = predict_on_model(img)
        elif isinstance(f, list):
            img = [cv2.imread(os.path.join(root_dir, f_[0])) for f_ in f]
            if multi_image:
                # The multi image handling responsibility is passed to the predict_on_model function here
                x_classify, x_count, x_count_total, x_classify_aux = predict_on_model(img)
            else:
                # In case the predict_on_model does not handle the multi_image scenario
                log_fn( "\n----{}".format(en(img)))
                # TODO: for some reason this sections has a memory leak, needs further investigation
                # Initializing the values to None
                x_classify, x_count, x_count_total, x_classify_aux = [None] * 4
                for im in img:
                    x_classify_, x_count_, x_count_total_, x_classify_aux_ = predict_on_model(im)
                    try:
                        x_classify += x_classify_
                    except:
                        x_classify = x_classify_

                    if x_count_ is not None:
                        try:
                            x_count += x_count_
                        except:
                            x_count = x_count_

                    if x_count_total_ is not None:
                        try:
                            x_count_total += x_count_total_
                        except:
                            x_count_total = x_count_total_
                x_classify /= len(img)
                if x_count_ is not None:
                    x_count /= len(img)
                if x_count_total_ is not None:
                    x_count_total /= len(img)
                x_classify_, x_count_, x_count_total_, x_classify_aux_ = [None] * 4
        #transformed_img = i[0].cuda()

        # When the model outputs the count as well, calculate the count related metrics
        count = None
        if x_count_total is not None:
            count = x_count_total.round_().squeeze().data.item()#.cpu().numpy()
            count_total_average_temp = int(count) == \
                sum([item_count for _, item_count in gt_labels.items()])
            count_total_average += count_total_average_temp
            # In the case where the models per item count metrics are also tracked
            if x_count is not None:
                count = get_adjusted_count(x_count, x_count_total, x_classify, prob_threshold)

        #log_fn(x_classify)

        # converting the gt_labels to use the string values of the class instead of the encoded integer value
        # TODO: the non class case needs more care
        try:
            if calc_negative_class:
                gt_labels = {classes[int(l)]:c for l,c in gt_labels.items() if int(l) in classes}#set(gt_labels.keys())
            else:
                gt_labels = {classes[int(l)]:c for l,c in gt_labels.items()}
        except KeyError:
            log_fn(row, classes, gt_labels)
            raise
        except:
            if 'None' in gt_labels:
                gt_labels = {'None':0}
            raise

        # Extracting the values for the class based metrics
        result = []
        result_gt_labels = []
        for i in range(total_classes_count):
            if x_classify[0,i] > prob_threshold:
                try:
                    result.append([classes[i],
                                   str(x_classify[0,i].detach().cpu().numpy())])
                                   # str(x_count[0,i].round().detach().cpu().numpy()),
                                   # str(count[0,i].item())])
                except KeyError:
                    result.append(["wrong-label",
                                   str(x_classify[0,i].detach().cpu().numpy())])
                                   # str(x_count[0,i].round().detach().cpu().numpy()),
                                   # str(count[0,i].item())])
            try:
                # Tracking the labels that went wrong
                if classes[i] in gt_labels and x_classify[0,i] <= prob_threshold:
                    result_gt_labels.append([classes[i] ,
                                             str(x_classify[0,i].detach().cpu().numpy())])#,
                                   #           str(x_count[0,i].round().detach().cpu().numpy()),
                                   # str(count[0,i].item())])
            except KeyError:
                pass
        # log_fn(result, gt_labels, "\n")
        # log_fn(set([n for n,a,b in result]) == set(gt_labels.keys()))

        # The results were used for historical reasons. Now extracting only the classes predicted.
        result_labels = set([n for n, _#,a,b,c
                             in result])

        # When handling the no class cases
        if len(result_labels) == 0:
            result_labels = set(['None'])

        # This makes sense only if there is a label expected to be predicted.
        if 'None' not in gt_labels:
            # Filter the combinations that are relevent to the ground truth labels
            current_combination_keys = [k for k in combinations_performance.keys() for k_1, k_2 in itertools.combinations(gt_labels.keys(), 2) if k_1 in k and k_2 in k]

            # I have no idea why this is here!!
            # key_to_comb_mapping = {}
            # for k in gt_labels.keys():
            #     for comb in current_combination_keys:
            #         if k in comb:
            #             try:
            #                 key_to_comb_mapping[k].append(comb)
            #             except:
            #                 key_to_comb_mapping[k] = [comb]

            # Increment the combinations relevent to the ground truth labels.
            for comb in current_combination_keys:
                combinations_performance[comb][1] +=1
                if comb[0] in result_labels and comb[1] in result_labels:
                    combinations_performance[comb][0] +=1

        # intersection and union between predicted labels and ground truth labels
        intersection = result_labels.intersection(gt_labels)
        union = result_labels.union(gt_labels)
        #log_fn(intersection, union)

        # Used when displaying the images of that predicted wrongly.
        count_success = True

        # Calculating the metrics for each label.
        # Have to iterate through all the labels cz calculating the false negative, etc.
        for label in used_labels:
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            #log_fn(gt_labels, result_labels,label)
            # Updating the counts values in macro_matrics
            if label in gt_labels:
                macro_matrics[label][4][1] += 1
                if count is not None:
                    if count[0, classes_reversed[label]] == gt_labels[label]:
                        macro_matrics[label][4][0] += 1
                    else:
                        count_success = False
        
            if label in gt_labels and label in result_labels:
                tp = 1
            elif label in gt_labels and label not in result_labels:
                fn = 1
            elif label not in gt_labels and label in result_labels:
                fp = 1
            else:
                tn = 1

            recall, precission, accuracy_confusion, accuracy_confusion_non_tn  = confusion_matrix_results(tp, fp, tn, fn)
            #log_fn(recall, precission, accuracy_confusion, accuracy_confusion_non_tn)
            micro_matrics[label][0] += tp
            micro_matrics[label][1] += fp
            micro_matrics[label][2] += tn
            micro_matrics[label][3] += fn

            if recall is not None:
                macro_matrics[label][0][0] += recall
                macro_matrics[label][0][1] += 1
            if precission is not None:
                macro_matrics[label][1][0] += precission
                macro_matrics[label][1][1] += 1
            macro_matrics[label][2] += accuracy_confusion
            if accuracy_confusion_non_tn is not None:
                macro_matrics[label][3][0] += accuracy_confusion_non_tn
                macro_matrics[label][3][1] += 1

        exact_match += 1 if all([l in result_labels for l in gt_labels]) \
            and all([l in gt_labels for l in results_label]) \
            else 0
        hemming += (len(union) - len(intersection))/used_labels_length
        accuracy += len(intersection)/len(gt_labels)
        if show_wrongs:
            if len(intersection)/len(gt_labels) != 1 or (not count_total_average_temp or not count_success):
                cv2.imshow("", img)
                log_fn("\n {}\n {}\n {}\n {}".format(gt_labels, result_labels, result, result_gt_labels))
                cv2.waitKey()
        # log_fn(x_classify.sort(descending = True))
        # max_index = x_classify.sort(descending = True)[1][0].cpu().numpy()
        # log_fn(max_index[0], "asdfasdfasdasdfasdfas")
        # log_fn([model_spec['data_codes_mapping'][idx] for idx in max_index])
        calced += 1
        # x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # x = cv2.resize(x, (7,35))
        #cv2.imshow("", img)

        #cv2.waitKey()
        # i_ += 1 
        # if i_ > 10:
        #     break

    return classes, combinations_performance, calced, exact_match, hemming, accuracy,\
        count_total_average, macro_matrics, micro_matrics
        
def eval_model(json_in, predict_on_model, root_dir, total_classes_count, prob_threshold = 0.1, calc_negative_class = False, multi_image = False, show_wrongs = False):
    assert torch_available
    assert cv2_available
    log_fn = log_with_temp_file("/tmp/mlp_torch_util_eval_model_output.txt", log)
    classes_, used_labels, predict_on_model = predict_on_model(multi_image = multi_image)
    classes, combinations_performance, calced, exact_match, hemming, accuracy,\
    count_total_average, macro_matrics, micro_matrics = _eval_model(log_fn, classes_, used_labels, json_in, predict_on_model, root_dir,
                                                                    total_classes_count, prob_threshold, calc_negative_class, multi_image, show_wrongs)
    log_fn("Used labels: {}".format(used_labels))
    log_fn("Classes: {}".format(classes))
    log_fn("Number of combinations: {}".format(len(combinations_performance)))
    log_fn("Number of images being evaled: {}".format(len(json_in)))
    log_fn("\nSKIPPED: {}".format(len(json_in) - calced))
    for k,v in micro_matrics.items():
        micro_matrics[k] = [x if x is not None else 0 for x in confusion_matrix_results(*v)]

    count_sparse_average = sum([v[4][0] for k,v in macro_matrics.items()])/ sum([v[4][1] for k,v in macro_matrics.items()])
    count_total_average = count_total_average / calced
    for k, v in macro_matrics.items():
        try:
            macro_matrics[k][0] = macro_matrics[k][0][0]/macro_matrics[k][0][1]
        except ZeroDivisionError:
            macro_matrics[k][0] = 0

        try:
            macro_matrics[k][1] = macro_matrics[k][1][0]/macro_matrics[k][1][1]
        except ZeroDivisionError:
            macro_matrics[k][1] = 0
        macro_matrics[k][2] = macro_matrics[k][2]/calced
        try:
            macro_matrics[k][3] = macro_matrics[k][3][0]/macro_matrics[k][3][1]
        except ZeroDivisionError:
            macro_matrics[k][3] = 0

        try:
            macro_matrics[k][4] = macro_matrics[k][4][0]/macro_matrics[k][4][1]
        except ZeroDivisionError:
            macro_matrics[k][4] = 0
            
    label_based_matrics = "Lable based metrics:\n"
    if calc_negative_class:
        used_labels += ['None']
    for label in used_labels:
        label_based_matrics += "\t{}: \n".format(label)

        label_based_matrics += "\t\tRecall:\t\t{:.4f}\t{:.4f}\n".format(micro_matrics[label][0],
                                                                      macro_matrics[label][0])
        
        label_based_matrics += "\t\tPrecision:\t{:.4f}\t{:.4f}\n".format(micro_matrics[label][1],
                                                                         macro_matrics[label][1])
        
        label_based_matrics += "\t\tAccuracy:\t{:.4f}\t{:.4f}\n".format(micro_matrics[label][2],
                                                                      macro_matrics[label][2])
        label_based_matrics += "\t\tAcc non tn:\t{:.4f}\t{:.4f}\n".format(micro_matrics[label][3],
                                                                      macro_matrics[label][3])
        label_based_matrics += "\t\tCount: \t{}\t{:.4f}\n".format(None,
                                                                  macro_matrics[label][4])
    log_fn("exact_match: {}".format(exact_match/calced))
    log_fn("hemming:     {}".format(hemming/calced))
    log_fn("accuracy:    {}".format(accuracy/calced))
    log_fn("count total: {}".format(count_total_average))
    log_fn("count sparse:{}".format(count_sparse_average))
    log_fn(label_based_matrics)
    for comb, vals in combinations_performance.items():
        try:
            log_fn("{} :{}".format(comb, vals[0]/vals[1]))
        except ZeroDivisionError:
            log_fn("{} : Irrelevent".format(comb))
    return log_fn.file_path

def get_adjusted_count(count, count_total, class_out, prob_threshold = 0.1):
    calced_counts  = (count * count_total)#.round()
    return_value = calced_counts#torch.zeros_like(calced_counts)
    return_value[class_out > prob_threshold] = torch.clamp(return_value[class_out > prob_threshold], min = 1).round()
    if count_total.round().sum().eq(return_value.sum()).item() == 0:
        np.set_printoptions(precision = 3, suppress = True)
    return return_value

def confusion_matrix_results(tp,fp, tn, fn):
    '''
return recall, prceision, accuracy, accuracy without counting the true
    negeative
'''
    try:
        recall = tp/(tp+fn)
    except ZeroDivisionError:
        recall = None
    try:
        precesion = tp/(tp+fp)
    except ZeroDivisionError:
        precesion = None

    try:
        acc_non_tn = tp/(tp+fp+fn)
    except ZeroDivisionError:
        acc_non_tn = None

    try:
        acc = (tp+tn)/(tp+fp+tn+fn)
    except:
        acc = 0
    return recall, precesion, acc, acc_non_tn
    
