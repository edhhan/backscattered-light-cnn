from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.cnn import CNN
from utils.data_utils import preprocess
from utils.training_utils import get_accuracy, train


def testing(nn_model, model_name, nb_photon, nb_epoch, lr, batch_size, GPU=False):
    """
    :param nn_model:
    :param nb_epoch:
    :param lr:
    :param GPU:
    :return:
    """
    best_precision = 0
    optimizer = optim.Adam(nn_model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    training_losses = []
    accuracies = []
    validation_losses = []

    train_loader, validation_loader, accuracy_loader = preprocess(model_name, nb_photon, batch_size)

    for epoch in tqdm(range(1, nb_epoch + 1)):

        # Train model
        nn_model = train(nn_model, train_loader, optimizer, GPU)

        # Accuracy on training set
        train_precision, loss_training = get_accuracy(nn_model, train_loader, GPU)
        training_losses.append(loss_training)

        # Accuracy and precision on validation set
        precision, loss_validation = get_accuracy(nn_model, validation_loader, GPU)
        validation_losses.append(loss_validation)
        accuracies.append(precision)
        if precision > best_precision:
            best_precision = precision

        # Scheduler
        scheduler.step(loss_validation)

    return best_precision, accuracy_loader, training_losses, validation_losses, accuracies


############
#   Main   #
############
GPU = False

input_photon = input("Nombre de photons:")

# Input nb of photons
input_ok = False
while input_ok is False:
    if input_photon == '1e3' or '1e3' or '1e5' or '1e6':
        input_ok = True
    else:
        input_model = input("wrong input of photon: {1e3, 1e4, 1e5, 1e6}")


nn_model = CNN()

# best set-up for CNN
lr = 0.00001
batch_size = 20

if input_photon == "1e3":
    nb_epoch = 20
elif input_photon == "1e4":
    nb_epoch = 25
elif input_photon == "1e5":
    nb_epoch = 75
elif input_photon == "1e6":
    nb_epoch = 300

precision, accuracy_loader, training_losses, validation_losses, accuracies = testing(nn_model,
                                                                                     "cnn",
                                                                                     input_photon,
                                                                                     nb_epoch,
                                                                                     lr,
                                                                                     batch_size,
                                                                                     GPU)
print('Final precision: {}'.format(get_accuracy(nn_model, accuracy_loader, GPU)))

plt.figure(1)
plt.plot(range(1, nb_epoch+1), training_losses, '--b', label='Training')
plt.plot(range(1,nb_epoch+1), validation_losses, '--r', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Log négatif de vraisemblance moyenne')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(range(1, nb_epoch+1), accuracies)
plt.ylabel('Precision (%)')
plt.xlabel('Epoch')
plt.show()
