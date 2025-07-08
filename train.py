import os
import sys

main_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(main_dir)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import model.pointnet as network
import dataset

SEED = 1204
torch.random.manual_seed(SEED)
np.random.seed(SEED)

def accuracy(preds, target):
    preds = torch.argmax(preds, dim=1)
    correct = torch.sum(preds == target)
    return correct.item()

def train_fn(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader,
             optimizer: torch.optim.Optimizer, scheduler, 
             criterion: torch.nn.Module, accuracy, train_records: dict,
             num_epochs, device, savedir = None):
    
    print("Started the model training...")
    for epoch in range(num_epochs):
        
        # train step on training set
        model.train()
        train_loss = 0
        correct_count = 0
        total_count = 0
        for points, labels in tqdm(train_loader, total=len(train_loader), desc = "running train loop..."):
            points, labels = points.to(device), labels.to(device)
            preds, matA, matB = model(points)
            loss = criterion(preds, labels, matA, matB)
            train_loss += loss.item()
            correct_count += accuracy(preds, labels)
            total_count += preds.shape[0]

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Save training records
        train_loss = train_loss/len(train_loader)
        train_acc = correct_count/total_count * 100
        train_records["train_loss"].append(train_loss)
        train_records["train_acc"].append(train_acc)

        # test step on test set
        model.eval()
        test_loss = 0
        correct_count = 0
        total_count = 0
        with torch.no_grad():
            for points, labels in tqdm(test_loader, total=len(test_loader), desc = "running test loop..."):
                points, labels = points.to(device), labels.to(device)
                preds, matA, matB = model(points)
                loss = criterion(preds, labels, matA, matB)
                test_loss += loss.item()
                correct_count += accuracy(preds, labels)
                total_count += preds.shape[0]
            
            test_loss /= len(test_loader)
            test_acc = correct_count/total_count * 100
            train_records["test_loss"].append(test_loss)
            train_records["test_acc"].append(test_acc)
        
        # print some of the records
        print(f"Epoch: {epoch+1}/{num_epochs} | train loss: {train_loss:.4f} | \
               test_loss: {test_loss:.4f} | train acc: {train_acc:.4f} | test_acc: {test_acc:.4f}")
        
        scheduler.step()
        if savedir and (epoch+1)%5 == 0:
            save_path = os.path.join(savedir, f"checkpoint_{epoch+1}.pth")
            save_dict = {
                "model": model.state_dict(),
                "train_records": train_records
            }
            torch.save(save_dict, save_path)


def generate_plots(train_records, save_dir = None):
    train_loss = train_records["train_loss"]
    train_acc = train_records["train_acc"]
    test_loss = train_records["test_loss"]
    test_acc = train_records["test_acc"]

    fig, ax = plt.subplots(1,2, figsize = (12, 8))

    ax[0].plot(train_loss, c="blue", label = "train loss")
    ax[0].plot(test_loss, c = "green", label = "test loss")
    ax[0].set_title("Loss Plot")
    ax[0].legend()

    ax[1].plot(train_acc, c="blue", label = "train accuracy")
    ax[1].plot(test_acc, c = "green", label = "test accuracy")
    ax[1].set_title("Accuracy Plot (%)")
    ax[1].legend()

    #plt.legend()
    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, "loss_acc_plot.png")
        plt.savefig(save_path)
    else:
        plt.show()

    
def test_fn(model: torch.nn.Module, test_loader: DataLoader, 
             criterion: torch.nn.Module, accuracy,
             device, checkpoint_dir=None, savedir = None):
    
    if checkpoint_dir:
        load_dict = torch.load(checkpoint_dir, weights_only=False)
        model.load_state_dict(load_dict["model"])
        train_records = load_dict["train_records"]

    test_loss = 0
    correct_count = 0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for points, labels in tqdm(test_loader, total=len(test_loader), desc = "running test loop..."):
            points, labels = points.to(device), labels.to(device)
            preds, matA, matB = model(points)
            loss = criterion(preds, labels, matA, matB)
            test_loss += loss.item()
            correct_count += accuracy(preds, labels)
            total_count += preds.shape[0]
        
        test_loss /= len(test_loader)
        test_acc = correct_count/total_count * 100
    
    # Now plot the train and test losses and accuracy
    if checkpoint_dir:
        generate_plots(train_records=train_records, save_dir=savedir)

    return test_acc


if __name__ == "__main__":
    print(main_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = True
    test = True

    # Define some paths
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/ModelNet10/ModelNet10"))
    checkpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    # points = torch.rand(8, 2048, 3).to(device)
    # label = torch.randint(0, 10, size=(8,)).to(device)
    # model = network.PointNet(num_classes=10).to(device)
    # out, matA, matB = model(points)
    # print(out.shape, matA.shape, matB.shape)

    # Get dataset and dataloader
    train_transform = dataset.get_train_transform()
    test_transform = dataset.get_test_transform()
    train_dataset = dataset.ModelNet(dataroot=dataroot, transform=train_transform, n_points=1024, dset="train")
    test_dataset = dataset.ModelNet(dataroot=dataroot, transform=train_transform, n_points=1024, dset="test")

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    print("length of train dataset: ", len(train_loader))
    print("length of test dataset: ", len(test_loader))

    # Define Model and loss function
    model = network.PointNet(num_classes=10).to(device)
    criterion = network.LossFunc(lamda=0.0001).to(device)
    # points, labels = next(iter(train_loader))
    # points, labels = points.to(device), labels.to(device)
    # print(points.dtype, labels.dtype, points.shape, labels.shape)
    # preds, matA, matB = model(points)
    # print(preds.shape, matA.shape, matB.shape)
    # loss = criterion(preds, labels, matA, matB)
    # print(loss)

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Now train the model
    num_epochs = 30
    train_records = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    #generate_plots(train_records=train_records)

    # train the model
    if train:
        train_fn(model = model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                accuracy=accuracy,
                train_records=train_records,
                num_epochs=num_epochs,
                device=device,
                savedir=checkpt_dir)

    # test the saved model
    if test:
        test_acc = test_fn(model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    accuracy=accuracy,
                    device=device,
                    checkpoint_dir=None,
                    savedir=None)
        print("Test Accuracy: ", np.round(test_acc,4))