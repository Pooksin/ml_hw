# -*- coding: utf-8 -*-
#mport argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import CifarNet as cfn

#parser = argparse.ArgumentParser()
#parser.add_argument('--test', action='store_true')
#parser.add_argument('--show_data', action='store_true')

# global data
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DATA_PATH = './data'
MODEL_PATH = './cifar_net.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    learning_rate = 0.001
    train_dataloader = cifar10_dataloader(train=True, 
        batch_size=16, shuffle=True, num_workers=6)
    
    model = cfn._cifarnet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9,weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    test_dataloader = cifar10_dataloader(train=False,batch_size=16, shuffle=True, num_workers=6)
    for epoch in range(50):
        train(train_dataloader, model, criterion, optimizer, epoch)
        torch.save(model.state_dict(), MODEL_PATH)
        print(epoch)
        test(test_dataloader, model, show_data=False)

#   test_dataloader = cifar10_dataloader(train=False,batch_size=16, shuffle=True, num_workers=6)
    test(test_dataloader, model, show_data=False)
#    for param in model.parameters():
#      print(param.data)
#    for var_name in optimizer.state_dict():
#      print(var_name, "\t", optimizer.state_dict()[var_name])
    
def cifar10_dataloader(root=DATA_PATH, train=True, transform=None, 
	shuffle=False, download=True, batch_size=4, num_workers=0):

	if transform is None:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

	dataset = torchvision.datasets.CIFAR10(root=root, 
		train=train, transform=transform, download=download)
	
	dataloader = torch.utils.data.DataLoader(dataset, 
		batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

	return dataloader 

def train(dataloader, model, criterion, optimizer, epoch):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        logit = model(images)
        loss = criterion(logit, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i%2000 == 1999:
            print(running_loss/2000.0)
            running_loss = 0.0

def test(dataloader, model, show_data):   
    if show_data:
        dataiter = iter(dataloader)
        images, labels = dataiter.next()
        # show images
        imshow(torchvision.utils.make_grid(images))  
        output = model(images.to(device))
        _, predicted = torch.max(output, 1)
        print('GT', ' '.join('%6s' % classes[labels[j]] for j in range(4)))
        print('PT', ' '.join('%6s' % classes[predicted[j]] for j in range(4)))
        print()
        
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def show_data(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%10s' % classes[labels[j]] for j in range(4)))

def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np    

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    main()
