import torch
import time
import wandb


class Experiment:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, writer,
                 epochs=1,
                 lr=0.001):
        self.model = model
        self.train_data = train_loader
        self.val_data = val_loader
        self.test_data = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.writer = writer

    def __train(self, train_data, lr_scheduler=None):
        self.model.train()
        train_loss = 0.0
        train_corrects = 0
        for inputs, targets in train_data:
            inputs.to(self.device)
            targets.to(self.device)

            # make the target value as a columns instead of one row
            targets = targets.view(targets.shape[0], 1)

            # Reset Gradients
            self.optimizer.zero_grad()

            # Forward Pass
            predicted_targets = self.model(inputs)

            # Calculate Loss
            loss = self.criterion(predicted_targets, targets)

            # Backward Pass
            loss.backward()

            # Update Weights
            self.optimizer.step()

            # loss.item() is batch loss, so we add them up to calculate train_loss
            train_loss += loss.item()

            # Adding following code additionally
            predictions, _ = torch.max(predicted_targets, 1)
            predictions = torch.round(predictions)
            train_corrects += (predictions == targets.squeeze(1)).sum().item()

        # schedule the learning rate for the next epoch of training if it has set before
        if lr_scheduler:
            lr_scheduler.step()

        return train_loss, train_corrects

    def __validate(self, val_data):
        self.model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, targets in val_data:
                inputs.to(self.device)
                targets.to(self.device)
                targets = targets.view(targets.shape[0], 1)  # make the target value as a columns instead of one row
                predicted_targets = self.model(inputs)
                loss = self.criterion(predicted_targets, targets)
                val_loss += loss.item()

                predictions, _ = torch.max(predicted_targets, 1)
                predictions = torch.round(predictions)
                val_corrects += (predictions == targets.squeeze(1)).sum().item()

        return val_loss, val_corrects

    def __test(self, test_data):
        n_correct = 0

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in test_data:
                inputs.to(self.device)
                targets.to(self.device)

                predicted_targets = self.model(inputs)

                # _ = value, index of classes (between 0 and 4)
                # _, predictions = torch.max(predicted_targets, 1)

                # the following is for when we use sigmoid and binary data with Binary Cross Entropy
                predictions, _ = torch.max(predicted_targets, 1)
                predictions = torch.round(predictions)

                # the following is for when we use softmax and regular data
                # add one element-wise to prediction in order to align with 1 to 5 as rate numbers
                # predictions = torch.add(predictions, 1)

                n_correct += (predictions == targets).sum().item()

            accuracy = n_correct / (len(test_data) * test_data.batch_size)
            if wandb.run is not None:
                wandb.log({'Test Acc': accuracy})
            print('Test Accuracy: %', 100 * accuracy)

    def __display_results(self, train_loss, train_corrects, val_loss, val_corrects, epoch):
        avg_train_loss = train_loss / len(self.train_data)
        train_acc = (train_corrects / (len(self.train_data) * self.train_data.batch_size))

        avg_val_loss = val_loss / len(self.val_data)
        val_acc = (val_corrects / (len(self.val_data) * self.val_data.batch_size))

        if (epoch + 1) % 10 == 0:
            print('Epoch: {} - Train Loss: {:.6f}, Validation Loss: {:.6f} - '
                  'Train ACC: {:.6f}, Validation ACC: {:.6f}, Learning Rate: {}'.format(
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                train_acc,
                val_acc,
                self.optimizer.param_groups[0]["lr"]
            ))

        self.writer.add_scalar('Training Loss', avg_train_loss, epoch)
        self.writer.add_scalar('Training Accuracy', train_acc, epoch)
        self.writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        self.writer.add_scalar('Validation Accuracy', val_acc, epoch)

        # self.writer.add_graph(self.model, self.train_data)

        if wandb.run is not None:
            wandb.log({'Train_Loss': avg_train_loss, 'Train Acc': train_acc,
                   'Validation Los': avg_val_loss, 'Validation Acc': val_acc})

    def run(self, lr_scheduler=None, verbose=True):

        start_time = time.time()

        print('-------- Training --------')
        self.model.to(self.device)

        # Training Loop
        for epoch in range(self.epochs):
            # Training
            train_loss, train_corrects = self.__train(self.train_data, lr_scheduler)

            # Validating
            val_loss, val_corrects = self.__validate(self.val_data)

            self.__display_results(train_loss, train_corrects, val_loss, val_corrects, epoch)

        # Testing
        print('-------- Testing --------')
        self.__test(self.test_data)

        wandb.finish()

        print('-------- Finished --------')
        print('Runtime: ', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
