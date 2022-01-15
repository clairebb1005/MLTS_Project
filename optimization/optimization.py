import datetime
import torch
import numpy as np
from matplotlib import pyplot as plt


class Optimization:
    def __init__(self, model, loss, optimizer, early_stopping_patience=20):
        self._cuda = torch.cuda.is_available()
        print(f"self._cuda={self._cuda}")
        self.device = torch.device("cuda" if self._cuda else "cpu")
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loss = []
        self.validation_loss = []
        self.valid_loss_min = np.inf
        self._early_stopping_patience = early_stopping_patience
        self.best_epoch = None

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions, run our forward pass of the model
        y_hat = self.model(x)

        # Pytorch accumulates gradients, we need to clear them out each time
        self.optimizer.zero_grad()

        # Computes loss
        loss = self.loss(y, y_hat)

        # Computes gradients
        loss.backward()

        # Updates parameters
        self.optimizer.step()

        # Returns the loss
        return loss.item()

    def save_checkpoint(self, epoch):
        torch.save({'state_dict': self.model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        print(f"Load the model of epoch = {epoch_n} for testing")
        ckp = torch.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self.model.load_state_dict(ckp['state_dict'])

    # allows the network's weights to be updated
    def train(self, train_loader, val_loader, epochs=30):
        # iterate through each epoch
        for epoch in range(epochs):
            batch_train_losses = []
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader, 0):
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                # train the model and calculate the gradient and loss
                train_loss = self.train_step(x_batch, y_batch)
                # append the loss of each batch
                batch_train_losses.append(train_loss)

            # append the mean of batch losses of current epoch
            training_loss = np.mean(batch_train_losses)
            self.train_loss.append(training_loss)

            validation_loss = self.evaluation(val_loader)

            if epoch % 10 == 0:
                print(
                    f"[Epoch:{epoch}/{epochs}], "
                    f"Training loss: {training_loss:.4f}\t "
                    f"Validation loss: {validation_loss:.4f}"
                )

            # EarlyStop:
            # If the validation loss does not decrease after a specified number of epochs,
            # then the training process will be stopped.
            # check whether the current loss is lower than the previous best value.
            if validation_loss < self.valid_loss_min:
                self.valid_loss_min = validation_loss
                self.best_epoch = epoch
                print('Validation loss decreased ({:.6f} --> {:.6f}). Save the model of current best epoch {} '.format(self.valid_loss_min, validation_loss, self.best_epoch))
                # save only the model that improve the loss
                self.save_checkpoint(epoch)

            # if not, count up for how long there was no progress
            else:
                self._early_stopping_patience -= 1

            # check whether early stopping should be performed and stop if so
            if self._early_stopping_patience < 0:
                print(f"Execute early_stopping and return the best_epoch ={self.best_epoch}")
                return self.best_epoch

        return self.best_epoch

    def evaluation(self, val_loader):
        # no need to calculate the gradients, just want to see the scores after this training epoch
        with torch.no_grad():
            batch_val_losses = []
            for batch_idx, (x_val, y_val) in enumerate(val_loader, 0):
                # load a batch of data to gpu
                x_val = x_val.to(device=self.device)
                y_val = y_val.to(device=self.device)
                # turn model into evaluation mode
                self.model.eval()
                y_hat = self.model(x_val)
                val_loss = self.loss(y_val, y_hat).item()
                batch_val_losses.append(val_loss)
            validation_loss = np.mean(batch_val_losses)
            self.validation_loss.append(validation_loss)
            return validation_loss

    def test(self, test_loader, best_epoch):
        # load the best model from training
        self.restore_checkpoint(epoch_n=best_epoch)
        # Here we don't need to train, so the code is wrapped in torch.no_grad()
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.to(device=self.device)
                y_test = y_test.to(device=self.device)
                # turn model into evaluation mode
                self.model.eval()
                # prediction of test data
                y_hat = self.model(x_test)
                predictions.append(y_hat.cpu().detach().numpy())
                values.append(y_test.cpu().detach().numpy())
        return predictions, values

    def plot_loss(self):
        plt.plot(self.train_loss, label="Training loss of each epoch")
        plt.plot(self.validation_loss, label="Validation loss of each epoch")
        plt.title("Loss")
        plt.savefig('output/plot_loss.png')