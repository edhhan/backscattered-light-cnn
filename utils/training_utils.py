import torch.nn.functional as F
from torch.autograd import Variable
import torch


def train(nn_model, loader, optimizer, GPU):
    """
    :param nn_model:
    :param loader:
    :param optimizer:
    :param GPU:
    :return:
    """
    nn_model.train()
    nn_model = nn_model.float()
    loss_training = 0

    for train_index, (data, target) in enumerate(loader):
        if not GPU:
            data, target = Variable(data), Variable(target)

        else:
            data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()

        optimizer.zero_grad()
        out_values = nn_model(data.float())

        target_nll = torch.zeros(len(target), dtype=torch.long)
        for i, one_hot in enumerate(target):
            for j, values in enumerate(one_hot[0][0]):
                if values == 1:
                    target_nll[i] = j

        loss = F.nll_loss(out_values, target_nll)

        #loss = F.multilabel_soft_margin_loss(out_values, target)

        loss_training = loss_training + loss.item()
        loss.backward()
        optimizer.step()

    return nn_model


def get_accuracy(nn_model, loader, GPU):
    """
    :param nn_model:
    :param loader:
    :param GPU:
    :return:
    """
    nn_model.eval()

    loss_validation = 0
    nb_correct = 0

    for data, target in loader:
        if not GPU:
            data, target = Variable(data), Variable(target)
        else:
            data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda()

        out_values = nn_model(data)
        target_nll = torch.zeros(len(target), dtype=torch.long)
        for i, one_hot in enumerate(target):
            for j, values in enumerate(one_hot[0][0]):
                if values == 1:
                    target_nll[i] = j
        loss_validation += F.nll_loss(out_values, target_nll, size_average=False).item()
        prediction = out_values.data.max(1, keepdim=True)[1]
        nb_correct += prediction.eq(target_nll.data.view_as(prediction)).cpu().sum()

    return nb_correct.item() * 100 / len(loader.dataset), loss_validation / len(loader.dataset)
