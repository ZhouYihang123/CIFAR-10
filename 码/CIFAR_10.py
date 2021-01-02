import torch
import torch.nn as zyhnet
import torch.nn.functional as F_hang
import torch.optim as best
import torchvision as he
import torchvision.transforms as hang_tr
import matplotlib.pyplot as plt
import numpy as np

#%%
class hang_str(zyhnet.Module):

    def __init__(mydefine):
        super(hang_str, mydefine).__init__()
        mydefine.hangcv1 = zyhnet.Conv2d(3, 6, 5)
        mydefine.hangcv2 = zyhnet.Conv2d(6, 16, 5)
        mydefine.hangfull1 = zyhnet.Linear(16 * 5 * 5, 120) 
        mydefine.hangfull2 = zyhnet.Linear(120, 84)
        mydefine.hangfull3 = zyhnet.Linear(84, 10)

    def forward(mydefine, para):
        para = F_hang.max_pool2d(F_hang.relu(mydefine.hangcv1(para)), (2, 2))
        para = F_hang.max_pool2d(F_hang.relu(mydefine.hangcv2(para)), 2)
        para = para.view(-1, mydefine.num_flat_features(para))
        para = F_hang.relu(mydefine.hangfull1(para))
        para = F_hang.relu(mydefine.hangfull2(para))
        para = mydefine.hangfull3(para)
        return para

    def num_flat_features(mydefine, para):
        size = para.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


hang_box = hang_str()
print(hang_box)

hang_trsize=64
hang_tesize=64


def load_data():
    hang_change=hang_tr.Compose(
        [hang_tr.ToTensor(),
         hang_tr.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    hangtr_set=he.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform =hang_change
        )
    hangtr_loader=torch.utils.data.DataLoader(
    hangtr_set,
    batch_size=hang_trsize,
    shuffle=True,
    num_workers=0)
    hangte_set=he.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=hang_change
    )
    hangte_loader=torch.utils.data.DataLoader(
    hangte_set,
    batch_size=hang_tesize,
    shuffle=False,
    num_workers=0)
    print("ready to train")
    return hangtr_loader,hangte_loader


hangtr_loader,hangte_loader=load_data()
hang_speed=0.01
hang_momentum=0.92
criterion=zyhnet.CrossEntropyLoss()
bestizer=best.SGD(hang_box.parameters(),lr=hang_speed,momentum=hang_momentum,weight_decay=0)

final_train_accu=[]
final_test_accu=[]
hang_time=2
for epoch in range(hang_time): #train data
    ruzyhneting_loss=0.0
    for i,data in enumerate(hangtr_loader,0):
        inputs,labels=data
        bestizer.zero_grad()
        outputs=hang_box(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        bestizer.step()
        ruzyhneting_loss+=loss.item()
        if i % 501==500:
            print('[%d,%5d] loss:%.3f' %
                  (epoch+1,i+1,ruzyhneting_loss/501))
            ruzyhneting_loss=0.0
    test_right=0
    allnumber=0
    inter=0
    store_train=[]
    with torch.no_grad():#calculate test accuracy
        for data in hangte_loader:
            images,labels=data
            outputs=hang_box(images)
            _,predicted=torch.max(outputs.data,1)
            allnumber +=labels.size(0)
            test_right +=(predicted==labels).sum().item()
            accu=100*test_right/allnumber
            store_train.append(accu)
        for elem in store_train:
            inter=inter+elem
        accuracy=inter/len(store_train)
        final_test_accu.append(accuracy)
    print('Accuracy of test :',accuracy)
    train_right=0
    allnumber=0
    inter=0
    store_test=[]
    with torch.no_grad():
        for data in hangtr_loader:#calculate train accuracy
            images,labels=data
            outputs=hang_box(images)
            _,predicted=torch.max(outputs.data,1)
            allnumber +=labels.size(0)
            train_right +=(predicted==labels).sum().item()
            accu=100*train_right/allnumber
            store_test.append(accu)
        for elem in store_test:
            inter=inter+elem
        accuracy=inter/len(store_test)
        final_train_accu.append(accuracy)
    print('Accuracy of train :',accuracy)

epo_iter=np.array(range(1,hang_time+1))
plt.plot(epo_iter,final_train_accu, 'r-', epo_iter, final_test_accu, 'g-')
    

