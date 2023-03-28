from datetime import datetime
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
import pickle
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES']="6,7"
device_ids = [0,1]
num_classes = 100

def read_pkl():
    # f = open(r'./pedcc_center/40_512.pkl', 'rb')
    # f = open(r'./pedcc_center/2_64_s.pkl', 'rb')
    # f = open(r'./pedcc_center/200_512.pkl','rb')
    # f = open(r'./pedcc_center/190_512.pkl', 'rb')
    f = open(r'./pedcc_center/100_512.pkl', 'rb')
    # f = open(r'./pedcc_center/101_512.pkl', 'rb')
    # f = open(r'./pedcc_center/51_512.pkl', 'rb')

    # f = open(r'./pedcc_center/102_512.pkl', 'rb')
    a = pickle.load(f)
    f.close()
    return a

def read_pkl2():
    # f = open(r'./pedcc_center/50_512.pkl', 'rb')
    # f = open(r'./pedcc_center/2_64_s.pkl', 'rb')
    f = open(r'./pedcc_center/200_512.pkl','rb')
    # f = open(r'./pedcc_center/100_512.pkl','rb')
    # f = open(r'./pedcc_center/100_512_anti.pkl', 'rb')
    a = pickle.load(f)
    f.close()
    return a

class AMSoftmax(nn.Module):
    def __init__(self, scale, margin, is_amp=False):
        super(AMSoftmax, self).__init__()
        self.scale = scale
        self.margin = margin
        self.is_amp = is_amp
    def forward(self, input, target):
        # self.it += 1
        cos_theta = input
        # let target reshape as (batchsize,1), here we get (0,4,34,...)
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        # here we set the index to convert the label info to a one-hot type
        index.scatter_(1, target.data.view(-1, 1), 1)
        # map the tensor to a byte type  WHY? obsolet
        index = index.bool() #index = index.byte()
        # the oppsite index
        index1 = ~index


        output = cos_theta * 1.0  # size=(B,Classnum)
        # for x in output[index]:
        #     if x > 0:
        #         x = x**3
        #     else:
        #         x = x**(1/3)

        # subtract the input tensor with hyperparameter margin
        output[index] = output[index] - self.margin

        # for x in output[index1]:
        #     if x > 0:
        #         x = x**(1/3)
        #     else:
        #         x = x**3

        if self.is_amp:
            output[index1] = output[index1] + self.margin
        output = output * self.scale


        logpt = F.log_softmax(output,dim=None)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

class CConvarianceLoss(nn.Module):
    def __init__(self, map_PEDCC):
        super(CConvarianceLoss, self).__init__()
        self.map_PEDCC = map_PEDCC
        return
    def forward(self, feature, label):
        average_feature = self.map_PEDCC[label.long().data].float().cuda()
        feature = l2_norm(feature)
        feature = feature - average_feature
        new_covariance100 = 1 / (feature.shape[0] - 1) * torch.mm(feature.T, feature).float()
        covariance100 = new_covariance100
        covariance100_loss = torch.sum(pow(covariance100, 2)) - torch.sum(pow(torch.diagonal(covariance100), 2))
        covariance100_loss = covariance100_loss / (covariance100.shape[0] - 1)
        return covariance100_loss, covariance100

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class CosineLinear_PEDCC(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear_PEDCC, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=False)
        #self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        map_dict = read_pkl()
        tensor_empty = torch.Tensor([]).cuda()
        for label_index in range(self.out_features):
            label_index = label_index
            tensor_empty = torch.cat((tensor_empty, map_dict[label_index].float().cuda()), 0)
        label_40D_tensor = tensor_empty.view(-1, self.in_features).permute(1, 0)
        label_40D_tensor = label_40D_tensor.cuda()
        self.weight.data = label_40D_tensor
        #print(self.weight.data)

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # weights normed
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)  x.dot(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)  #
        cos_theta = cos_theta.clamp(-1, 1)
        cos_theta = cos_theta * xlen.view(-1, 1)

        return cos_theta  # size=(B,Classnum,1)

def myl2_norm(input):                          # According to amsoftmax, we have to normalize the feature, which is x here
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def get_acc_pre(output, label):
    total = output.shape[0]
    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    output = myl2_norm(output)
    output_pedcc = PEDCCout(output)
    _, pred_label = output_pedcc.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

def train_pretrained_first(step, net, train_data, valid_data, num_epochs, criterion, criterion1, criterion02, modelname=None):
    LR = 0.1
    val_acc=0
    top_acc=30

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    net = net.eval()

    prev_time = datetime.now()
    map_dict = read_pkl()
    map_dict2 = read_pkl2()
    small_amount = 0.05
    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    # PEDCC access here
    for epoch in range(num_epochs):

        if epoch in [0, 30, 60, 90]:
            if epoch != 0:
                LR *= 0.1

            optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):# for data, index_ in tqdm(train_data):
            if torch.cuda.is_available():
                label = label
                label1 = label
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label1:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()), 0)   # Define the PEDCC as our label
                label_mse_tensor = tensor_empty.view(-1, 512)       #(-1, dimension)
                label_mse_tensor = label_mse_tensor.cuda()

            output = net(im)  # the output to do classification
            output1 = myl2_norm(output)
            output2 = PEDCCout(output1)
            loss1 = criterion(output2, label)
            loss2 = criterion1(output1, label_mse_tensor) * 512

            loss = loss1 + loss2 #+ 0.2 * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc_pre(output, label)

        print("small_amount")
        print(small_amount)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    label = label
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc_pre(output, label)
            val_acc=(valid_acc / len(valid_data))
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, Train Loss3: %f, Train Loss4: %f "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data), train_loss3 / len(train_data), train_loss4 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(modelname + '.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        print(val_acc)
        if val_acc>top_acc:
            top_acc=val_acc
            torch.save(net,  str(top_acc) + 'ACC.pkl')
        if epoch == 100:
            torch.save(net.module, modelname + str(epoch) + '_epoch.pth')

def train_pretrained_PEDCC(step, net, train_data, valid_data, num_epochs, criterion, criterion1, criterion02,
                           modelname=None):
    LR = 0.1
    val_acc = 0
    top_acc = 30

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    net = net.eval()

    prev_time = datetime.now()
    map_dict = read_pkl()
    map_dict2 = read_pkl2()
    small_amount = 0.05
    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    # PEDCC access here
    for epoch in range(num_epochs):

        if epoch in [0, 30, 60, 90]:
            if epoch != 0:
                LR *= 0.1

            params = [
                {'params': net.module.layer1.parameters(), 'lr': LR * 0.1},
                {'params': net.module.layer2.parameters(), 'lr': LR * 0.1},
                {'params': net.module.layer3.parameters(), 'lr': LR * 0.2},
                {'params': net.module.layer4.parameters(), 'lr': LR * 0.2},
                {'params': net.module.fc.parameters()}, ]
            optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):  # for data, index_ in tqdm(train_data):
            if torch.cuda.is_available():
                label = label
                label1 = label
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label1:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()),
                                             0)  # Define the PEDCC as our label
                label_mse_tensor = tensor_empty.view(-1, 512)  # (-1, dimension)
                label_mse_tensor = label_mse_tensor.cuda()

            output = net(im)  # the output to do classification
            output1 = myl2_norm(output)
            output2 = PEDCCout(output1)
            loss1 = criterion(output2, label)
            loss2 = criterion1(output1, label_mse_tensor) * 512
            loss = loss1 + loss2  # + 0.2 * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc_pre(output, label)

        print("small_amount")
        print(small_amount)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    label = label
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc_pre(output, label)
            val_acc = (valid_acc / len(valid_data))
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, Train Loss3: %f, Train Loss4: %f "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data),
                       train_loss3 / len(train_data), train_loss4 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(modelname + '.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        print(val_acc)
        if val_acc > top_acc:
            top_acc = val_acc
            torch.save(net, str(top_acc) + 'ACC.pkl')
        if epoch == 100:
            torch.save(net.module, modelname + str(epoch) + '_epoch.pth')

def train_pretrained_PEDCCpure(step, net, train_data, valid_data, num_epochs, criterion, criterion1, criterion02,
                               modelname=None):
    LR = 0.1
    val_acc = 0
    top_acc = 30

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    net = net.eval()

    prev_time = datetime.now()
    map_dict = read_pkl()
    map_dict2 = read_pkl2()
    small_amount = 0.05
    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    # PEDCC access here
    for epoch in range(num_epochs):

        if epoch in [0, 30, 60, 90]:
            if epoch != 0:
                LR *= 0.1

            params = [
                {'params': net.module.layer1.parameters(), 'lr': LR * 0.2},
                {'params': net.module.layer2.parameters(), 'lr': LR * 0.2},
                {'params': net.module.layer3.parameters(), 'lr': LR * 0.5},
                {'params': net.module.layer4.parameters(), 'lr': LR * 0.7},
                {'params': net.module.fc.parameters()}, ]
            optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):  # for data, index_ in tqdm(train_data):
            if torch.cuda.is_available():
                label = label
                label += 20 * (step - 1)
                label1 = label
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label1:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()),
                                             0)  # Define the PEDCC as our label
                label_mse_tensor = tensor_empty.view(-1, 512)  # (-1, dimension)
                label_mse_tensor = label_mse_tensor.cuda()

            output = net(im)  # the output to do classification
            output1 = myl2_norm(output)
            output2 = PEDCCout(output1)
            loss1 = criterion(output2, label)
            loss2 = criterion1(output1, label_mse_tensor) * 512
            loss = loss1 + loss2  # + 0.2 * loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc_pre(output, label)

        print("small_amount")
        print(small_amount)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    label = label
                    # label += 10 * (step - 1)
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc_pre(output, label)
            val_acc = (valid_acc / len(valid_data))
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, Train Loss3: %f, Train Loss4: %f "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data),
                       train_loss3 / len(train_data), train_loss4 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(modelname + '.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        print(val_acc)
        if val_acc > top_acc:
            top_acc = val_acc
            torch.save(net, str(top_acc) + 'ACC.pkl')
        if epoch == 100:
            # torch.save(net.module.state_dict(), modelname+str(epoch)+'_epoch.pth')
            torch.save(net.module, modelname + str(epoch) + '_epoch.pth')

def train_pretrained_He(step, net, train_data, valid_data, num_epochs, criterion, criterion1, criterion02,
                        modelname=None):
    LR = 0.1
    val_acc = 0
    top_acc = 30

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()

    net = net.eval()

    prev_time = datetime.now()
    map_dict = read_pkl()
    map_dict2 = read_pkl2()
    small_amount = 0.05
    # PEDCC access here
    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    for epoch in range(num_epochs):

        if epoch in [0, 30, 60, 90]:
            if epoch != 0:
                LR *= 0.1
            layer1_params = list(map(id, net.module.layer1.parameters()))
            layer2_params = list(map(id, net.module.layer2.parameters()))
            layer3_params = list(map(id, net.module.layer3.parameters()))
            layer4_params = list(map(id, net.module.layer4.parameters()))
            base_params = filter(lambda p: id(p) not in layer1_params + layer2_params + layer3_params + layer4_params,
                                 net.parameters())
            params2 = [{'params': base_params},
                       {'params': net.module.layer1.parameters(), 'lr': LR * 0.2},
                       {'params': net.module.layer2.parameters(), 'lr': LR * 0.2},
                       {'params': net.module.layer3.parameters(), 'lr': LR * 0.5},
                       {'params': net.module.layer4.parameters(), 'lr': LR * 0.7}, ]
            if step == 1:
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
            else:
                optimizer = optim.SGD(params2, lr=LR, momentum=0.9, weight_decay=5e-4)

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_acc = 0
        net = net.train()
        for im, label in tqdm(train_data):  # for data, index_ in tqdm(train_data):
            if torch.cuda.is_available():
                label = label
                label1 = label
                im = im.cuda()  # (bs, 3, h, w)
                label = label.cuda()  # (bs, h, w)
                tensor_empty = torch.Tensor([]).cuda()
                for label_index in label1:
                    tensor_empty = torch.cat((tensor_empty, map_dict[label_index.item()].float().cuda()),
                                             0)  # Define the PEDCC as our label
                label_mse_tensor = tensor_empty.view(-1, 512)  # (-1, dimension)
                label_mse_tensor = label_mse_tensor.cuda()

            output = net(im)  # the output to do classification
            output1 = myl2_norm(output)
            output2 = PEDCCout(output1)

            loss1 = criterion(output2, label)
            loss2 = criterion1(output1, label_mse_tensor) * 512  # the output to do the MSE loss

            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_acc += get_acc_pre(output, label)

        print("small_amount")
        print(small_amount)
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    label = label
                    # label += 10 * (step - 1)
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.data
                valid_acc += get_acc_pre(output, label)
            val_acc = (valid_acc / len(valid_data))
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, LR: %f, Train Loss1: %f, Train Loss2: %f, Train Loss3: %f, Train Loss4: %f "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data), LR, train_loss1 / len(train_data), train_loss2 / len(train_data),
                       train_loss3 / len(train_data), train_loss4 / len(train_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)
        f = open(modelname + '.txt', 'a+')
        f.write(epoch_str + time_str + '\n')
        f.close()
        print(val_acc)
        if val_acc > top_acc:
            top_acc = val_acc
            torch.save(net, str(top_acc) + 'ACC.pkl')
        if epoch == 100:
            # torch.save(net.module.state_dict(), modelname+str(epoch)+'_epoch.pth')
            torch.save(net.module, modelname + str(epoch) + '_epoch.pth')

def testimg_savefile(net1,dataset,filename):
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
        net1 = net1.cuda()
    net1.eval()
    map_dict = read_pkl()

    tensor_empty1 = torch.Tensor([])

    for label_index in range(num_classes):
        tensor_empty1 = torch.cat((tensor_empty1, map_dict[label_index].float()), 0)
    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    flag = 0
    for step, (im, label) in enumerate(tqdm(dataset)):  # Store the feature of training data

        if torch.cuda.is_available():
            im = im.cuda()  # (bs, 3, h, w)
        output1 = net1(im)#.cpu().detach().numpy()
        output1 = myl2_norm(output1)
        output1 = PEDCCout(output1).cpu().detach().numpy()
        if flag==0:
            alloutput = output1
        else:
            alloutput = np.vstack((alloutput, output1))

        flag = 1
    np.save(filename, alloutput)

def testimg_savefile_norm(net1,dataset,filename):
    if torch.cuda.is_available():
        net1 = torch.nn.DataParallel(net1, device_ids=device_ids)
        net1 = net1.cuda()

    net1.eval()
    map_dict = read_pkl()

    tensor_empty1 = torch.Tensor([])

    for label_index in range(num_classes):
        tensor_empty1 = torch.cat((tensor_empty1, map_dict[label_index].float()), 0)

    PEDCCout = CosineLinear_PEDCC(512, num_classes)
    flag = 0
    for step, (im, label) in enumerate(tqdm(dataset)):  # Store the feature of training data

        if torch.cuda.is_available():
            im = im.cuda()  # (bs, 3, h, w)

        with torch.no_grad():
            output1 = net1(im)#.cpu().detach().numpy()
            output_fornorm = output1.cpu().detach().numpy()
            mo = np.linalg.norm(output_fornorm, axis=1)
            mo = mo.reshape(-1, 1)
            output2 = myl2_norm(output1)
            output2 = PEDCCout(output2).cpu().detach().numpy()
            onestage = np.multiply(output2, mo)
        if flag==0:
            alloutput = onestage
        else:
            alloutput = np.vstack((alloutput, onestage))

        flag = 1
    np.save(filename, alloutput)
