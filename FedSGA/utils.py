import torch
import copy
import math


def organise_by_class(num_classes: int, data, device):
    indices_class = [[] for c in range(num_classes)]
    images = [torch.unsqueeze(data[i][0], dim=0) for i in range(len(data))]
    labels = [data[i][1] for i in range(len(data))]
    for i, lab in enumerate(labels):
        indices_class[lab].append(i)
    images = torch.cat(images, dim=0).to(device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return indices_class, images, labels


def plug(img, label):
    return [(x, y) for x, y in zip(img, label)]

def unlearn(device, dataloader, model, loss_fc, optimizer):
    size = len(dataloader.dataset)
    torch.save(model, '../tmp/tmp.pth')
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fc(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    recall_ptk = torch.load('../tmp/tmp.pth')
    model.load_state_dict(compute_sga_grad(recall_ptk, model))


def compute_sga_grad(recall_ptk, curr_net):
    final_state_dict = copy.deepcopy(recall_ptk.state_dict())
    pres_state_dict = recall_ptk.state_dict()
    curr_state_dict = curr_net.state_dict()
    for k in final_state_dict.keys():
        final_state_dict[k] += (pres_state_dict[k] - curr_state_dict[k])
    return final_state_dict