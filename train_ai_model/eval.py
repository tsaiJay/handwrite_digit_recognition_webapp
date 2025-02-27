import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms


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

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

    data, label = testset[0]
    # data = transform(data)  # when data is not coming from dataset(include transform)


    ''' Uncommand this section to show input data on screen '''
    # transform_to_show = transforms.ToPILImage()
    # img = transform_to_show(data)
    # img.show()


    ''' load model (structure include)'''
    net = torch.load('./weight/lenet.pt', map_location=data.device)


    ''' inference '''
    outputs = net(data.unsqueeze(0))
        
    _, pred = torch.max(outputs, dim=1)
    print(pred)


if __name__ == "__main__":
    main()