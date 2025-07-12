import os
import sys

main_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(main_dir)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

import model.pointnet as pointnet
import  model.pointnetpp_model as pointnet_pp
import dataset

SEED = 1204
torch.random.manual_seed(SEED)
np.random.seed(SEED)

def train_fn(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader,
             optimizer: torch.optim.Optimizer, scheduler, 
             criterion: torch.nn.Module, accuracy, train_records: dict,
             num_epochs, device, savedir = None):
    
    print("Started the model training...")
    best_val_acc = 0
    for epoch in range(num_epochs):
        
        # train step on training set
        model.train()
        train_loss = 0
        correct_count = 0
        total_count = 0
        for points, labels in tqdm(train_loader, total=len(train_loader), desc = "running train loop..."):
            points, labels = points.to(device), labels.to(device)
            preds = model(points)
            loss = criterion(preds, labels)
            train_loss += loss.item()
            correct_count += accuracy(preds, labels)
            total_count += labels.shape[0]

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
                preds = model(points)
                loss = criterion(preds, labels)
                test_loss += loss.item()
                correct_count += accuracy(preds, labels)
                total_count += labels.shape[0]
            
            test_loss /= len(test_loader)
            test_acc = correct_count/total_count * 100
            train_records["test_loss"].append(test_loss)
            train_records["test_acc"].append(test_acc)
        
        # print some of the records
        print(f"Epoch: {epoch+1}/{num_epochs} | train loss: {train_loss:.4f} | \
               test_loss: {test_loss:.4f} | train acc: {train_acc:.4f} | test_acc: {test_acc:.4f}")

        if scheduler:
            scheduler.step()

        # Save best model
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            save_path = os.path.join(savedir, "best_model.pth")
            save_dict = {
                "model": model.state_dict(),
                "train_records": train_records
            }
            torch.save(save_dict, save_path)

        # Save models after regular intervals
        if savedir and (epoch+1)%5 == 0:
            save_path = os.path.join(savedir, f"checkpoint_{epoch+1}.pth")
            save_dict = {
                "model": model.state_dict(),
                "train_records": train_records
            }
            torch.save(save_dict, save_path)

def load_model(model: torch.nn.Module, checkpt_path):
    load_dict = torch.load(checkpt_path, weights_only=False)
    model.load_state_dict(load_dict["model"])
    train_records = load_dict["train_records"]
    return train_records


def argument_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default = "pointnet")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--n_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--resume_from", type=int, default=None)
    
    return parser.parse_args()



if __name__ == "__main__":
    print(main_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argument_parser()

    # Get dataset and dataloader
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/ModelNet10/ModelNet10"))
    train_transform = dataset.get_train_transform()
    val_transform = dataset.get_test_transform()
    train_dataset = dataset.ModelNet(dataroot=dataroot, transform=train_transform, n_points=args.n_points, dset="train")
    val_dataset = dataset.ModelNet(dataroot=dataroot, transform=val_transform, n_points=args.n_points, dset="test")

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    print("length of train dataset: ", len(train_loader))
    print("length of test dataset: ", len(val_loader))

    # Define Model and loss function

    if args.model_name == "pointnet":
        model = pointnet.PointNet(num_classes=args.num_classes).to(device)
        criterion = pointnet.LossFunc(lamda=0.0001).to(device)
        accuracy = pointnet.accuracy
    
    elif args.model_name == "pointnet++":
        model = pointnet_pp.PointNetPlusPlus(num_classes=args.num_classes, in_channels=3).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        accuracy = pointnet_pp.accuracy
    
    else:
        raise NotImplementedError("The specified model doesn't exist")

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Now train the model
    train_records = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": []
    }

    #generate_plots(train_records=train_records)

    # train the model
    if args.resume_from:
        checkpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{args.resume_from}.pth")
        train_records = load_model(model, checkpt_path)

    train_fn(model = model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            accuracy=accuracy,
            train_records=train_records,
            num_epochs=args.num_epochs,
            device=device,
            savedir=args.checkpoint_dir)
