import numpy as np
import os as OO
import torch
import torch.optim as optim
import net

def runCNN(trainloader, device, optimizer, net, criterion, validloader, PATH, fileName, epochNum):
    trainLoss = []
    validationLoss = []

    for _ in range(epochNum):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / (i+1)
        trainLoss.append(epoch_loss)
        
        with torch.no_grad(): 
            running_loss = 0.0
            for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward 
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            epoch_loss = running_loss / (i+1)
            validationLoss.append(epoch_loss)

            torch.save(net.state_dict(), OO.path.join(PATH, "{}.pth".format(fileName)))

    return trainLoss, validationLoss

def runCNN_earlyStop(trainloader, device, optimizer, net, criterion, validloader, PATH, fileName, epochNum):
    best_loss = np.float('inf')
    trainLoss = []
    validationLoss = []
    best_epoch = 0
    range_epochs = 5

    for epoch in range(epochNum):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / (i+1)
        trainLoss.append(epoch_loss)
        
        with torch.no_grad(): 
            running_loss = 0.0
            for i, data in enumerate(validloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # forward 
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            epoch_loss = running_loss / (i+1)
            validationLoss.append(epoch_loss)

            # save the best model based on validation loss
            if epoch_loss < best_loss:
                torch.save(net.state_dict(), OO.path.join(PATH, "{}.pth".format(fileName)))
                best_loss = epoch_loss
                best_epoch = epoch

            # stop early if it has been several epochs since last best
            if (epoch - best_epoch) > range_epochs:
                break

    return trainLoss[0:best_epoch], validationLoss[0:best_epoch]

def testCNN(testloader, net, device):
    correct, total = 0, 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    return correct, total