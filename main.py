import os
from utils import *
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
num_classes = 100

from torchvision import models
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

map_dict = read_pkl()
map_PEDCC = torch.Tensor([])
for i in range(num_classes):
    map_PEDCC = torch.cat((map_PEDCC, map_dict[i].float()), 0)
map_PEDCC = map_PEDCC.view(num_classes, -1)  # (class_num, dimension)

map_dict2 = read_pkl2()
map_PEDCC2 = torch.Tensor([])
for i in range(num_classes):
    map_PEDCC2 = torch.cat((map_PEDCC2, map_dict2[i].float()), 0)
map_PEDCC2 = map_PEDCC2.view(num_classes, -1)  # (class_num, dimension)

criterion = AMSoftmax(7.5, 0.35, is_amp=False)
criterion1 = nn.MSELoss()
criterion0 = CConvarianceLoss(map_PEDCC)
criterion02 = CConvarianceLoss(map_PEDCC2)

def train_total():
    train_path = './tinyimagenet/total_train'
    train_data = torchvision.datasets.ImageFolder(train_path,
                                                  # './EMNIST/20+20/train_20_2nd', #'./cifar100/25/train_25_1st', # /total_train
                                                  transform=transforms.Compose([
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.Resize([224, 224]),
                                                      # transforms.RandomCrop(size=32, padding=4),
                                                      transforms.ToTensor(),
                                                      normalize,
                                                  ])
                                                  )
    test_path = './tinyimagenet/total_test'
    test_data = torchvision.datasets.ImageFolder(test_path,
                                                 # './EMNIST/20+20/test_20_2nd', #'./cifar100/25/test_25_1st', # total_test
                                                 transform=transforms.Compose([
                                                     # transforms.CenterCrop(size=32),
                                                     transforms.Resize([224, 224]),
                                                     transforms.ToTensor(),
                                                     normalize,
                                                 ]))

    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)


    model_name = './0206/0307ti_64_total_pre'
    cnn = torch.load('./barlowtwins-main/resnet.pth', map_location='cpu')
    # cnn = models.resnet50(pretrained=False)
    # cnn = models.resnet50(pretrained=True)
    # model_fp = os.path.join('./SimCLR/', "checkpoint_{}.tar".format(100))
    cnn.fc = nn.Linear(in_features=2048, out_features=512, bias=True)
    print(cnn)
    step = 2
    train_pretrained_first(step, cnn, # ("./25/3rd/withGood20_2/25_3rd_withGood20100_epoch.pkl",
                     train_loader, test_loader, 101, criterion, criterion1, criterion02, model_name)
    print(str(step),"has been trained successfully")




def train_manynets(steps, con):
    for i in range(steps):
        step = i+1
        train_path = './cifar100/20/train_20_'+str(step)
        train_data = torchvision.datasets.ImageFolder(train_path,
                                                      # './EMNIST/20+20/train_20_2nd', #'./cifar100/25/train_25_1st', # /total_train
                                                      transform=transforms.Compose([
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.Resize([224, 224]),
                                                          # transforms.RandomCrop(size=32, padding=4),
                                                          transforms.ToTensor(),
                                                          normalize,
                                                      ])
                                                      )
        test_path = './cifar100/20/test_20_'+str(step)
        test_data = torchvision.datasets.ImageFolder(test_path,
                                                     # './EMNIST/20+20/test_20_2nd', #'./cifar100/25/test_25_1st', # total_test
                                                     transform=transforms.Compose([
                                                         # transforms.CenterCrop(size=32),
                                                         transforms.Resize([224, 224]),
                                                         transforms.ToTensor(),
                                                         normalize,
                                                     ]))

        train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)


        model_name = './0206/0309cibarlow1122_64_20_'+str(step)
        cnn = torch.load('./barlowtwins-main/resnet.pth', map_location='cpu')
        # cnn = torch.load('./resnet-50-simclr1.pth')
        # cnn = models.resnet50(pretrained=False)
        # cnn = models.resnet50(pretrained=True)
        cnn.fc = nn.Linear(in_features=2048, out_features=512, bias=True)
        print(cnn)
        if con == True:
            if (step >1):
                model_name_last = './0206/0309cibarlow1122_64_20_'+str(step-1)+'100_epoch.pth'
                cnn = torch.load(model_name_last)
        train_pretrained_PEDCC(step, cnn,  # ("./25/3rd/withGood20_2/25_3rd_withGood20100_epoch.pkl",
                             train_loader, test_loader, 101, criterion, criterion1, criterion02, model_name)
        # train_pretrained_PEDCCpure(step, cnn,  # ("./25/3rd/withGood20_2/25_3rd_withGood20100_epoch.pkl",
        #                        train_loader, test_loader, 101, criterion, criterion1, criterion02, model_name)
        # train_pretrained_He(step, cnn,  # ("./25/3rd/withGood20_2/25_3rd_withGood20100_epoch.pkl",
        #                        train_loader, test_loader, 101, criterion, criterion1, criterion02, model_name)
        print(str(step),"has been trained successfully")



def save_output(steps):
    for i in range(steps):
        step = i+1
        test_path1 = './cifar100/20/test_20_5'
        test_data1 = torchvision.datasets.ImageFolder(test_path1,
                                                      # './cifar100/25/test_50_2nd', # '/home/data/HZK/fashionMNIST/test',        # tesing data
                                                      transform=transforms.Compose([
                                                          # transforms.CenterCrop(size=64),
                                                          transforms.Resize([224, 224]),
                                                          # transforms.Resize(size=32),
                                                          transforms.ToTensor(),
                                                          normalize,
                                                      ])
                                                      )

        test_loader1 = Data.DataLoader(dataset=test_data1, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        model_name = './0206/0309cibarlow1122_64_20_'+str(step)+'100_epoch.pth'
        cnn = torch.load(model_name)
        filename = './0206/test/0309cibarlow1122_64_20_'+str(step)+'.npy'
        #(cnn, test_loader1,filename)
        testimg_savefile_norm(cnn, test_loader1, filename)
        print(str(step),"has been tested successfully")

def final_test(steps):
    flag = 0
    for i in range(steps):
        step = i + 1

        filename = './0206/test/0309cibarlow1122_64_20_' + str(step) + '.npy'
        reloadtmp = np.load(filename)
        if flag == 0:
            output_together = reloadtmp
        else:
            output_together = np.hstack((output_together, reloadtmp))
        flag = 1


    train_path1 = './cifar100/20/test_20_5'
    test_data1 = torchvision.datasets.ImageFolder(train_path1,
                                                          # './cifar100/25/test_50_2nd', # '/home/data/HZK/fashionMNIST/test',        # tesing data
                                                          transform=transforms.Compose([
                                                              # transforms.CenterCrop(size=64),
                                                              transforms.Resize([224, 224]),
                                                              # transforms.Resize(size=32),
                                                              transforms.ToTensor(),
                                                              normalize,
                                                          ])
                                                          )

    test_loader1 = Data.DataLoader(dataset=test_data1, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    i = 0
    num_correct = 0
    sum = 0
    for step, (im, label) in enumerate(tqdm(test_loader1)):  # Store the feature of training data
        if torch.cuda.is_available():
            label = label
            im = im.cuda()  # (bs, 3, h, w)
            label = label.cuda()  # (bs, h, w)

        label1 = label.cpu().numpy()
        num = im.shape[0]

        prediction = output_together[sum:sum+num]
        sum += num
        prediction_label = prediction.argmax(axis=1) % num_classes
        num_correct += (prediction_label == label1).sum().item()
        ACC = num_correct / sum
        print(ACC)
        print(sum)
        i = i+1



steps = 5

train_manynets(steps, False)
save_output(steps)
final_test(steps)
# train_total()
