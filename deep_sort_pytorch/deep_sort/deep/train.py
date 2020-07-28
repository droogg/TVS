import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torch
from torch.utils.data import Dataset
import os
import glob
from pathlib import Path
import cv2
import torchvision
from PIL import Image
import numpy as np
import random

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args = parser.parse_args()

print("CUDA :", torch.cuda.is_available())
path = '/home/vic/PycharmProjects/DS_neuro/Task/detection/MY/detect_project/vric/vric_train.txt'
path_img = '/home/vic/PycharmProjects/DS_neuro/Task/detection/MY/detect_project/vric/train_images/'

# path_test = '/home/vic/PycharmProjects/DS_neuro/Task/detection/MY/detect_project/vric/vric_gallery.txt'
# path_img_test = '/home/vic/PycharmProjects/DS_neuro/Task/detection/MY/detect_project/vric/gallery_images/'

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


# def oreder_keys(ddict):
#     return {x: y for x,y in zip(range(len(ddict)), ddict.values())}

# sample_test_keys = []
# for item in set(d.values()):
#     keys = [x for x, y in d.items() if y == item]
#     num_of_el = int(len(keys)*0.2)
#     sample_test = random.choices(keys, k=num_of_el)
#     sample_test_keys.extend(sample_test)
# test_dict_tmp = {x: y for x, y in d.items() if x in sample_test_keys}
# train_dict_tmp = {x: y for x, y in d.items() if x not in sample_test_keys}
# test_l_tmp = {x:y for x, y in d2.items() if x in test_dict.keys()}
# train_l_tmp = {x:y for x, y in d2.items() if x in train_dict.keys()}

# test_dict = oreder_keys(test_dict_tmp)
# train_dict = oreder_keys(train_dict_tmp)
# test_l = oreder_keys(test_l_tmp)
# train_l = oreder_keys(train_l_tmp)

def load_data(path, path_img_dir):
    def file_exist(file_dict):
        for file in file_dict.values():
            if not os.path.isfile(file):
                return False
        return True
    try:
        path = str(Path(path))
        parent = str(Path(path).parent) + os.sep
        if os.path.isfile(path):
            with open(path, 'r') as f:
                f = f.read().splitlines()
                f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
                # if not f[0].startswith('./') or os.path.isfile(f[0])
        elif os.path.isdir(path):
            f = glob.iglob(path + os.sep + '*.*')
        else:
            raise Exception('%s does not exist' % path)
        img_cls_dict = {x: int(y) for x, y in enumerate(line.split()[1] for line in f)}
        img_file_path_dict = {x: ''.join((path_img_dir, y)) for x, y in enumerate(line.split()[0] for line in f)}
        # self.img_file = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
    except: raise Exception(f'Ошибка загрузки документа {path}')
    assert file_exist(img_file_path_dict), f'Нет файлов по указанному пути: {path_img_dir}'
    return img_cls_dict, img_file_path_dict


def train_test_separator(cls_dict, path_dict):
    def oreder_keys(ddict):
        return {x: y for x, y in zip(range(len(ddict)), ddict.values())}

    def swap_dict(ddict):
        new_d = {}
        for key, val in ddict.items():
            if val in new_d:
                new_d[val].append(key)
            else:
                new_d[val] = [key]
        return new_d

    swaped_dict = swap_dict(cls_dict)
    sample_test_keys = []
    for item in set(cls_dict.values()):
        # keys = [x for x, y in cls_dict.items() if y == item]
        keys = swaped_dict[item]
        num_of_el = int(len(keys) * 0.2)
        sample_test = random.choices(keys, k=num_of_el)
        sample_test_keys.extend(sample_test)
    test_cls_dict_tmp = {x: y for x, y in cls_dict.items() if x in sample_test_keys}
    train_cls_dict_tmp = {x: y for x, y in cls_dict.items() if x not in sample_test_keys}
    test_path_dict_tmp = {x: y for x, y in path_dict.items() if x in test_cls_dict_tmp.keys()}
    train_path_dict_tmp = {x: y for x, y in path_dict.items() if x in train_cls_dict_tmp.keys()}

    test_cls_dict = oreder_keys(test_cls_dict_tmp)
    train_cls_dict = oreder_keys(train_cls_dict_tmp)
    test_path_dict = oreder_keys(test_path_dict_tmp)
    train_path_dict = oreder_keys(train_path_dict_tmp)
    return train_cls_dict, train_path_dict, test_cls_dict, test_path_dict


class LoadData(Dataset):
    def __init__(self, img_obj_dict, img_file_path_dict, transform=None):
        self.img_obj = img_obj_dict
        self.img_file_path = img_file_path_dict
        self.transform = transform
        posible_keys = set(img_obj_dict.values())
        self.cls_trg_conf = {key: val for key, val in zip(posible_keys, range(len(posible_keys)))}

    def __len__(self):
        return len(self.img_obj)

    def __getitem__(self, index):
        img = self.load_image(index)
        cls_obj = self.img_obj[index]
        target = self.cls_trg_conf[cls_obj]
        if self.transform:
            img = self.transform(img)
        return img, target

    def load_image(self, index):
        img_path = self.img_file_path[index]
        img = Image.open(img_path)
        assert img is not None, 'Image Not Found ' + img_path
        return img


# img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']

# class LoadData(Dataset):
#     def __init__(self, path, path_img_dir, transform=None):
#         try:
#             path = str(Path(path))
#             parent = str(Path(path).parent) + os.sep
#             if os.path.isfile(path):
#                 with open(path, 'r') as f:
#                     f = f.read().splitlines()
#                     f = [x.replace('./', parent) if x.startswith('./') else x for x in f]
#                     # if not f[0].startswith('./') or os.path.isfile(f[0])
#             elif os.path.isdir(path):
#                 f = glob.iglob(path + os.sep + '*.*')
#             else:
#                 raise Exception('%s does not exist' % path)
#             self.img_obj = {x: int(y) for x, y in enumerate(line.split()[1] for line in f)}
#             self.img_file_path = {x: ''.join((path_img_dir, y)) for x, y in enumerate(line.split()[0] for line in f)}
#             # self.img_file = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
#         except:
#             raise Exception(f'Ошибка загрузки документа {path}')
#         assert self.file_exist(self.img_file_path), f'Нет файлов по указанному пути: {path_img_dir}'
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.img_obj)
#
#     def __getitem__(self, index):
#         img = self.load_image(index)
#         target = torch.tensor(self.img_obj[index])
#         if self.transform:
#             img = self.transform(img)
#         return img, target
#
#
#     def load_image(self, index):
#         img_path = self.img_file_path[index]
# #         img = cv2.imread(img_path)[:,:,::-1] # to RGB
#         img = Image.open(img_path)
#         assert img is not None, 'Image Not Found ' + img_path
#         return img
#
#     @staticmethod
#     def file_exist(file_dict):
#         for file in file_dict.values():
#             if not os.path.isfile(file):
#                 return False
#         return True


# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root,"train")
test_dir = os.path.join(root,"test")
# transform_train = torchvision.transforms.Compose([
#     torchvision.transforms.RandomCrop((128,64),padding=4),
#     torchvision.transforms.RandomHorizontalFlip(),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# transform_test = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((128,64)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# разделение на траин и тест
img_obj_dict, img_file_path_dict = load_data(path, path_img)
train_cls_dict, train_path, test_cls_dict, test_path = train_test_separator(img_obj_dict, img_file_path_dict)


dataset = LoadData(train_cls_dict, train_path, transform=transform_train)
testset = LoadData(test_cls_dict, test_path, transform=transform_test)
# trainloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
#     batch_size=64,shuffle=True
# )
trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,shuffle=True
)
# testloader = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
#     batch_size=64,shuffle=True
# )
# num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))
num_classes = max(len(set(dataset.img_obj.values())), len(set(testset.img_obj.values())))
# num_classes = max(dataset.img_obj.values())

# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if args.resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        # print(inputs, labels)
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(start_epoch, start_epoch+40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%20==0:
            lr_decay()


if __name__ == '__main__':
    main()
