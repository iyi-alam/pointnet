import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import model.pointnet as pointnet
import model.pointnetpp_model as pointnet_pp
import dataset
import argparse
import numpy as np
from cad_visualizer import plot_3d_grid


def load_model(model: torch.nn.Module, checkpt_path):
    load_dict = torch.load(checkpt_path, weights_only=False)
    model.load_state_dict(load_dict["model"])
    train_records = load_dict["train_records"]
    return train_records

def generate_plots(train_records, save_dir = None):
    train_loss = train_records["train_loss"]
    train_acc = train_records["train_acc"]
    test_loss = train_records["test_loss"]
    test_acc = train_records["test_acc"]

    fig, ax = plt.subplots(1,2, figsize = (12, 4))

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
    # else:
    #     plt.show()

    
def test_fn(model: torch.nn.Module, test_loader: DataLoader, 
             accuracy, device, checkpoint_dir=None, savedir = None):
    
    if checkpoint_dir:
        train_records = load_model(model, checkpoint_dir)

    correct_count = 0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for points, labels in tqdm(test_loader, total=len(test_loader), desc = "running test loop..."):
            points, labels = points.to(device), labels.to(device)
            #breakpoint()
            preds = model(points)
            correct_count += accuracy(preds, labels)
            total_count += labels.shape[0]
            #print(f"Correct: {correct_count}/{total_count}")
        
        test_acc = correct_count/total_count * 100
    
    # Now plot the train and test losses and accuracy
    if checkpoint_dir:
        generate_plots(train_records=train_records, save_dir=savedir)

    return test_acc

def plot_preds(model: torch.nn.Module, test_dset: dataset.ModelNet, device, checkpoint_dir=None, savedir = None, plot_name = None):
    if checkpoint_dir:
        train_records = load_model(model, checkpoint_dir)

    total_files = len(test_dset)
    random_idx = np.random.choice(list(range(total_files)), replace=False, size=32)

    file_paths = []
    pcs = []
    labels = []

    for idx in random_idx:
        file_paths.append(test_dset.file_paths[idx])
        pc, label = test_dset[idx]
        pcs.append(pc)
        labels.append(label)
    
    pcs = torch.stack(pcs, dim=0)
    labels = torch.stack(labels, dim = 0)
    with torch.no_grad():
        pcs, labels = pcs.to(device), labels.to(device)
        preds = model(pcs)
    
    #print(pointnet.accuracy(preds, labels))

    if isinstance(preds, tuple):
        preds = torch.argmax(preds[0], dim=-1)
    else:
        preds = torch.argmax(preds, dim=-1) # for pointnet++ model
    titles = [f"Target: {test_dset.key_to_obj[labels[i].item()]}\nPredicted: {test_dset.key_to_obj[preds[i].item()]}" for i in range(32)]
    # print(titles)
    # print(preds, labels)
    idxes = list(range(32))
    np.random.shuffle(idxes)
    file_paths = [file_paths[i] for i in idxes[:8]]
    titles = [titles[i] for i in idxes[:8]]
    
    plot_3d_grid(file_paths=file_paths, titles=titles, savedir=savedir, plot_name=plot_name)
    

    

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default = None)
    parser.add_argument("--model_name", type=str, default="pointnet")
    parser.add_argument("--num_classes", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argument_parser()

    # Get dataset and dataloader
    dataroot = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/ModelNet10/ModelNet10"))
    test_transform = dataset.get_test_transform()
    test_dataset = dataset.ModelNet(dataroot=dataroot, transform=test_transform, n_points=1024, dset="test")
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    print("length of test dataset: ", len(test_loader))

    if args.model_name == "pointnet":
        model = pointnet.PointNet(num_classes=args.num_classes).to(device)
        accuracy = pointnet.accuracy
    
    elif args.model_name == "pointnet++":
        model = pointnet_pp.PointNetPlusPlus(num_classes=args.num_classes, in_channels=3).to(device)
        accuracy = pointnet_pp.accuracy
    
    else:
        raise NotImplementedError("The specified model doesn't exist")

    # test_acc = test_fn(model = model,
    #                    test_loader=test_loader,
    #                    accuracy=accuracy,
    #                    checkpoint_dir=args.checkpoint_path,
    #                    savedir=args.output_dir,
    #                    device = device)
    
    # print("Accuracy on test set: ", test_acc)

    # Get a sample prediction plot
    plot_preds(model=model, test_dset=test_dataset, device=device, checkpoint_dir=args.checkpoint_path, savedir=args.output_dir, plot_name=args.model_name)