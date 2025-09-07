import torch
def evaluate(model, dataloader,device, criterion):
    correct = 0
    total = 0
    sum_of_losses_per_epoch = 0
    number_of_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            number_of_batches += 1 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            sum_of_losses_per_epoch += loss.item()

    accuracy = correct / total
    loss = sum_of_losses_per_epoch / number_of_batches
    return accuracy,loss

def evaluateGoogleNet(model, dataloader,device,criterion):
    correct = 0
    total = 0
    sum_of_losses_per_epoch = 0
    number_of_batches = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            number_of_batches += 1 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            sum_of_losses_per_epoch += loss.item()

    accuracy = correct / total
    loss = sum_of_losses_per_epoch / number_of_batches
    return accuracy,loss
