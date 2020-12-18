import numpy as np
import os as OO
import torch
import torch.optim as optim
import net

def runCNN(trainloader, device, optimizer, net, criterion, validloader, PATH, fileName, epochNum):
    best_loss = np.float('inf')
    trainLoss = []
    validationLoss = []
    newPath = PATH + "/{}_lr_{}".format(fileName, optimizer.defaults['lr'])
    OO.mkdir(newPath)

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
            savePath = OO.path.join(newPath, '{}_epoch_{}_lr_{}.pth'.format(fileName, epoch, optimizer.defaults['lr']))
            if fileName == 'adam':
                torch.save(net.state_dict(), savePath)
            elif fileName == 'sgd':
                torch.save(net.state_dict(), savePath)
            elif epoch_loss < best_loss:
                torch.save(net.state_dict(), OO.path.join(newPath, "{}_epoch_{}.pth".format(fileName, epoch)))
                best_loss = epoch_loss
            else:
                torch.save(net.state_dict(), OO.path.join(newPath, "{}_epoch_{}.pth".format(fileName, epoch)))

    return trainLoss, validationLoss

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

def createNets(num, netType):
    nets = []
    for _ in range(num):
        if netType is "sig":
            nets.append(net.SigNet())
        elif netType is "relu":
            nets.append(net.ReluNet())

    return nets

def createOptimizers(adamNets, sgdNets, learningRates):
    adamOptimizers = []
    sgdOptimizers = []

    for i in range(len(learningRates)):
        adamOptimizers.append(optim.Adam(adamNets[i].parameters(), learningRates[i]))
        sgdOptimizers.append(optim.SGD(sgdNets[i].parameters(), learningRates[i]))

    return adamOptimizers, sgdOptimizers

def runCNN_noStop(trainloader, device, optimizer, net, criterion, validloader, PATH, fileName, epochNum):
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

def testCNNwithConfusionMatrix(testloader, net, device, class_correct, class_total):
    correct, total = 0, 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            c = (predicted == labels.to(device))
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return correct, total, class_correct, class_total

def manyAccuracies(accuracy, testNet, kind, optimizers, loadNet, testLoader, learningRates, PATH, numTests, device):
    for i in range(len(learningRates)):    
        temp = []
        newPath = PATH + "/{}_lr_{}".format(kind, optimizers[i].defaults['lr'])
        for j in range(numTests):
            fileName = "{}_epoch_{}_lr_{}.pth".format(kind, j, learningRates[i])
            testNet.load_state_dict(torch.load(OO.path.join(newPath, fileName)))
            correct, total = testCNN(testLoader, testNet, device)
            temp.append(100 * correct / total)
        accuracy.append(temp)

def eachAccuracy(testNet, numTests, PATH, loadNet, testLoader, device):
    accuracy = []
    for j in range(numTests):
        testNet.load(PATH, "{}_epoch_{}.pth".format(loadNet, j))
        correct, total = testCNN(testLoader, testNet, device)
        accuracy.append(100 * correct / total)
    return accuracy