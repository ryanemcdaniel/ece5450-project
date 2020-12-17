import numpy as np
import os as OO
import torch
import torch.optim as optim
import net

def runCNN(trainloader, device, optimizer, net, criterion, validloader, PATH, fileName, epochNum):

    best_loss = np.float('inf')
    trainLoss = []
    validationLoss = []

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
            savePath = OO.path.join(PATH, '{}_epoch_{}_lr_{}.pth'.format(fileName, epoch, optimizer.defaults['lr']))
            if fileName == 'adam':
                torch.save(net.state_dict(), savePath)
            elif fileName == 'sgd':
                torch.save(net.state_dict(), savePath)
            elif epoch_loss < best_loss:
                torch.save(net.state_dict(), OO.path.join(PATH, "{}_epoch_{}.pth".format(fileName, epoch)))
                best_loss = epoch_loss
            else:
                torch.save(net.state_dict(), OO.path.join(PATH, "{}_epoch_{}.pth".format(fileName, epoch)))

    return trainLoss, validationLoss