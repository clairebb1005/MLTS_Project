import datetime
import torch
import numpy as np
from matplotlib import pyplot as plt


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.valid_loss_min = np.inf

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions, run our forward pass of the model
        y_hat = self.model(x)

        # Pytorch accumulates gradients, we need to clear them out each time
        self.optimizer.zero_grad()

        # Computes loss
        loss = self.loss_fn(y, y_hat)

        # Computes gradients
        loss.backward()

        # Updates parameters
        self.optimizer.step()

        # Returns the loss
        return loss.item()

    # allows the network's weights to be updated
    def train(self, train_loader, val_loader, batch_size=512, n_epochs=30, n_features=1):
        model_path = f'models/{self.model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
                y_batch = y_batch.to(self.device)
                # train the model and calculate the gradient and loss
                loss = self.train_step(x_batch, y_batch)
                # append the loss of each batch
                batch_losses.append(loss)
            # append the batch loss of each epoch
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            # no need to calculate the gradients, just want to see the scores after this training epoch
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # load the data to gpu
                    x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    y_hat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, y_hat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if epoch % 10 == 0:
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )

            # save only the model that improve the loss
            if validation_loss <= self.valid_loss_min:
                torch.save(self.model.state_dict(), model_path)
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.valid_loss_min,
                                                                                                validation_loss))
                self.valid_loss_min = validation_loss

    def evaluate(self, test_loader, batch_size=512, n_features=1):
        # Here we don't need to train, so the code is wrapped in torch.no_grad()
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                y_hat = self.model(x_test)  # prediction of testing
                predictions.append(y_hat.to(self.device).detach().numpy())
                values.append(y_test.to(self.device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss of each epoch")
        plt.plot(self.val_losses, label="Validation loss of each epoch")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()