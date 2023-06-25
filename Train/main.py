import torch
import os
import sys

from prepare_dataset import main_prepare_dataset
from train_func import train_one_epoch, evaluate
from model_module import Model1, Model4, Model2, Model3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESUME = False

FOLDER = 6

if __name__ == "__main__":

    # if sys.argv[1] == True:
    #     RESUME = False
    # if sys.argv[2] is not None:
    #     FOLDER = sys.argv[2]

    lr_scheduler = None
    num_classes = 3
    num_epochs = 3001
    iterations = 0
    start_epoch = 0

    # call out model
    if FOLDER == 1:
        model= Model1(num_classes)
    elif FOLDER == 2:
        model= Model2(num_classes)
    else:
        model= Model4(num_classes)

    print(DEVICE)
    # move model to the right device
    model.to(DEVICE)

    # get the model using our helper function
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0133, momentum=0.9, weight_decay=0.00004)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, last_epoch = start_epoch - 1 )


    if not os.path.exists('/home/locth/Documents/KLTN/Train/weight/model_' + f"{FOLDER}"):
        os.makedirs('/home/locth/Documents/KLTN/Train/weight/model_'+ f"{FOLDER}")

    data_loader, data_loader_test = main_prepare_dataset()

    if RESUME: 
        checkpoint = torch.load('/home/locth/Documents/KLTN/Train/weight/model_'+ f"{FOLDER}" +'/checkpoint.pth',map_location= DEVICE)
        #checkpoint = torch.load('/content/gdrive/MyDrive/Self_Driving_Car/weight/model_'+ f"{FOLDER}" +'/checkpoint.pth'',map_location= device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        iterations = checkpoint['iteration']

    for epoch in range(start_epoch, num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(iterations, model, optimizer, data_loader, DEVICE, epoch, print_freq=10, lr_scheduler=lr_scheduler)
        # update the learning rate
        lr_scheduler.step()

        #save func
        path = "/home/locth/Documents/KLTN/Train/weight/model_" + f"{FOLDER}" +"/"
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'iteration' : iterations
            }
        
        
        torch.save(checkpoint, path + "checkpoint.pth")
        if (epoch%100 ==0): torch.save(checkpoint, path + f"{epoch}.pth")


        # evaluate on the test dataset
        if (epoch%50 == 0):
            coco_evaluator = evaluate(model, data_loader_test, device=DEVICE)