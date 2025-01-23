import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms

from model import LeNet  # script model.py -> LeNet


def train(model, trainloader, criterion, optimizer, device):
    model.train()
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        ''' 1. forward '''
        outputs = model(inputs)

        ''' 2. calculate loss '''
        loss = criterion(outputs, labels)
        
        ''' 3. cumulate grad '''
        loss.backward()

        ''' 4. optimize weight '''
        optimizer.step()

        _, pred = torch.max(outputs, dim=1)
        correct = (pred == labels).sum()
        total = labels.size(0)
        acc = correct / total * 100
        if i % 100 == 0:
            print(f'train acc: {acc:3.1f}, loss: {loss.item():.3f}')


@torch.no_grad()
def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(testloader):

        inputs, labels = inputs.to( device), labels.to(device)
        outputs = model(inputs)
        
        _, pred = torch.max(outputs, dim=1)
        correct += (pred == labels).sum()
        total += labels.size(0)
    
    acc = correct / total * 100
    print(f'-->> test acc: {acc:.6f}')
    return acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('use device', device)

    batch_size = 16
    epoch_size = 5

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    lenet = LeNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lenet.parameters(), lr=0.01, momentum=0.9)

    best_acc = 0
    for epoch in range(epoch_size):
        
        train(lenet, trainloader, criterion, optimizer, device)
        acc = test(lenet, testloader, device)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(lenet, 'weight/lenet.pt')

    print('finish!!')


if __name__ == "__main__":
    main()
