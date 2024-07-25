import os
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR10
import numpy as np
import data_utils
from torch.utils.data import Subset, DataLoader
# import data_utils

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

class DirichletDataset(Dataset):
    def __init__(self, alpha, cifar_dataset):
        self.alpha = alpha
        self.cifar_dataset = cifar_dataset
        self.dirichlet_dist = torch.distributions.Dirichlet(alpha)

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        proportions = self.dirichlet_dist.sample()
        return {'image': image, 'label': label, 'proportions': proportions}


def Makedata(num_clients): # removed batch_size which was an argument
    
    # split data to 10 parts to save each partition
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    # trainloaders = []
    # valloaders = []
    for idx,ds in enumerate(datasets):
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        # trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        # valloaders.append(DataLoader(ds_val, batch_size=batch_size))
        torch.save(ds_train,f"../Data/iid/Train/train_data_{idx}.pth")
        torch.save(ds_val,f"../Data/iid/Validation/val_data_{idx}.pth")

def Makenoniid_one(num_clients): # removed batch_size which was an argument
    traindata=CIFAR10("./dataset", train=True, download=True, transform=transform)
    def seperator(data, labels):
        unique=np.unique(labels)
        tr_classes={}
        # val_classes={}
        for c in unique:
            indices = np.where(labels == c)
            idx = indices[0]
            idx = idx.reshape(1,-1)
            idx = idx.flatten().tolist()
            print(len(idx))
            tr_classes[c]=[]
            for i in idx:
                tr_classes[c].append(data[i])
        return tr_classes
    filter_dict= seperator(traindata, traindata.targets)
    data=[]
    for i in filter_dict.keys():
        for j in filter_dict[i]:
            data.append(j)
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),          
        transforms.ToTensor(),                  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = data
    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    print(lengths)
    # datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    datasets = [trainset[i*size:(i+1)*size] for i,size in enumerate(lengths)]
    print(datasets[0][0][0].shape)

    # Split each partition into train/val and create DataLoader
    for idx,ds in enumerate(datasets):
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        
        torch.save(ds_train,f"../Data/Non-iid-1/Train/train_data_{idx}.pth")
        torch.save(ds_val,f"../Data/Non-iid-1/Validation/val_data_{idx}.pth")

def Make_unbalancedDirichlet(num_clients, val_frac, dir, alpha):
    transform_tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=.40),
        transforms.RandomRotation(30),
        transforms.ToTensor(),           
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),           
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform_tr)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform_test)
    
    train_path = f"../Data/{dir}/Train"
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    
    val_path = f"../Data/{dir}/Validation"
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    
    seed=2023
    hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                     num_clients,
                                     balance=None,
                                     partition="dirichlet",
                                     unbalance_sgm=0.3,
                                     dir_alpha=alpha,
                                     seed=seed)
    div = hetero_dir_part.client_dict

    csv_file = f"../Data/{dir}/cifar10_hetero_dirichlet_{alpha}_clients-{num_clients}.csv"
    num_classes=10
    partition_report(trainset.targets, hetero_dir_part.client_dict,
                    class_num=num_classes,
                    verbose=False, file=csv_file)
    for id in div.keys():
        indxs=list(div[id])
        # print(indxs)
        # print(type(indxs))
        length = len(indxs)
        val_sample = int(length*val_frac)
        validation = random.sample(indxs, val_sample)
        train = [item for item in indxs if item not in validation]
        train_data = Subset(trainset, train)
        val_data = Subset(trainset, validation)
        torch.save(train_data,f"../Data/{dir}/Train/train_data_{id}.pth")
        torch.save(val_data,f"../Data/{dir}/Validation/val_data_{id}.pth")
    
    test_path = f"../Data/{dir}/Test"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    balanced_dir_part = CIFAR10Partitioner(testset.targets,
                                     num_clients,
                                     balance=True,
                                     partition="iid",   
                                     seed=seed)
    div_test = hetero_dir_part.client_dict

    csv_file_test = f"../Data/{dir}/Test/cifar10_balanced_test_clients-{num_clients}.csv"
    num_classes=10
    partition_report(testset.targets, balanced_dir_part.client_dict,
                    class_num=num_classes,
                    verbose=False, file=csv_file_test)
    for id in div_test.keys():
        indxs=list(div[id])
        test_data = Subset(testset, indxs)
        torch.save(test_data,f"../Data/{dir}/Test/test_data_{id}.pth")

def Make_short_unbalancedDirichlet(num_clients, val_frac, dir_name):

    directory=f"../Data/{dir_name}"

    if not os.path.exists(directory):
        os.makedirs(directory)
        train=os.path.join(directory,"Train")
        val=os.path.join(directory,"Validation")
        os.makedirs(train)
        os.makedirs(val)
        print(f"Directory '{directory}' created.")

    transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406],   # Mean of ImageNet dataset
        std=[0.229, 0.224, 0.225]     # Standard deviation of ImageNet dataset
    )
])
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    num_samples_per_class = 1000
    
    classes=trainset.classes
    label_dict = {} # ship : 8
    for class_name in classes:
        label_dict[class_name]=trainset.class_to_idx[class_name]      
    print(trainset.targets)
    print(classes)
    print(label_dict)
    idx_dict={} # 1:[234,3242,12,....]
    for class_name in label_dict:
        idx_dict[label_dict[class_name]]=[]

    for i in range(len(trainset)):
        current_class=trainset[i][1]
        idx_dict[current_class].append(i)

    for class_id in idx_dict:
        idx_dict[class_id]=idx_dict[class_id][:num_samples_per_class]
    
    train_dataset=[]
    for class_id in idx_dict:
        train_dataset+=idx_dict[class_id]
    print(train_dataset)
    print(len(train_dataset))
    train_data=Subset(trainset, train_dataset)

    subset_targets=[trainset.targets[i] for i in train_data.indices]

    seed=2023
    hetero_dir_part = CIFAR10Partitioner(subset_targets,
                                     num_clients,
                                     balance=None,
                                     partition="dirichlet",
                                     dir_alpha=0.3,
                                     seed=seed)
    div = hetero_dir_part.client_dict

    csv_file = f"../Data/{dir_name}/cifar10_hetero_dir_0.3_{num_clients}clients.csv"
    num_classes=10
    partition_report(trainset.targets, hetero_dir_part.client_dict,
                    class_num=num_classes,
                    verbose=False, file=csv_file)
    for id in div.keys():
        indxs=list(div[id])
        # print(indxs)  
        # print(type(indxs))
        length = len(indxs)
        val_sample = int(length*val_frac)
        validation = random.sample(indxs, val_sample)
        train = [item for item in indxs if item not in validation]
        train_data = Subset(trainset, train)
        val_data = Subset(trainset, validation)
        torch.save(train_data,f"../Data/{dir_name}/Train/train_data_{id}.pth")
        torch.save(val_data,f"../Data/{dir_name}/Validation/val_data_{id}.pth")


    
def prepare_data(dir_name):
    directory=f"../Data/{dir_name}"

    if not os.path.exists(directory):
        os.makedirs(directory)
        train=os.path.join(directory,"Train")
        val=os.path.join(directory,"Validation")
        os.makedirs(train)
        os.makedirs(val)
        print(f"Directory '{directory}' created.")
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path="../../../Latest/VRGAS/src/Simulator/data_FEDBN/MNIST", channels=1, percent=0.1, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="../../../Latest/VRGAS/src/Simulator/data_FEDBN/MNIST", channels=1, percent=0.1, train=False, transform=transform_mnist)
    torch.save(mnist_trainset,f"../Data/{dir_name}/Train/train_data_0.pth")
    torch.save(mnist_testset,f"../Data/{dir_name}/Validation/val_data_0.pth")

    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/SVHN', channels=3, percent=0.1,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/SVHN', channels=3, percent=0.1,  train=False, transform=transform_svhn)
    torch.save(svhn_trainset,f"../Data/{dir_name}/Train/train_data_1.pth")
    torch.save(svhn_testset,f"../Data/{dir_name}/Validation/val_data_1.pth")

    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/USPS', channels=1, percent=0.1,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/USPS', channels=1, percent=0.1,  train=False, transform=transform_usps)
    torch.save(usps_trainset,f"../Data/{dir_name}/Train/train_data_2.pth")
    torch.save(usps_testset,f"../Data/{dir_name}/Validation/val_data_2.pth")

    # Synth Digits
    synth_trainset     = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/SynthDigits/', channels=3, percent=0.1,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/SynthDigits/', channels=3, percent=0.1,  train=False, transform=transform_synth)
    torch.save(synth_trainset,f"../Data/{dir_name}/Train/train_data_3.pth")
    torch.save(synth_testset,f"../Data/{dir_name}/Validation/val_data_3.pth")

    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/MNIST_M/', channels=3, percent=0.1,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path='../../../Latest/VRGAS/src/Simulator/data_FEDBN/MNIST_M/', channels=3, percent=0.1,  train=False, transform=transform_mnistm)
    torch.save(mnistm_trainset,f"../Data/{dir_name}/Train/train_data_4.pth")
    torch.save(mnistm_testset,f"../Data/{dir_name}/Validation/val_data_4.pth")
    
    print(len(mnist_trainset),len(mnist_testset))
    print(len(svhn_trainset),len(svhn_testset))
    print(len(usps_trainset),len(usps_testset))
    print(len(synth_trainset),len(synth_testset))
    print(len(mnistm_trainset),len(mnistm_testset))
    # mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    # mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    # svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    # svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    # usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    # usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    # synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    # synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    # mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    # mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    # train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    # test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    # return train_loaders, test_loaders

if __name__=="__main__":
    nc=10
    # alpha=torch.tensor([1.0, 1.0, 1.0])
    dir="DIGITS"
    # Make_short_unbalancedDirichlet(nc, 0.2, dir)
    prepare_data(dir_name=dir)
