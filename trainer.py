import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.utils.prune as prune
from torch.amp import autocast, GradScaler

torch.manual_seed(0)

class Trainer():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def epoch_loop(self, epochs:int, model:nn.Module, optimizer:torch.optim,
                         criterion, loader:DataLoader, mixed_precision,
                         scheduler, is_train:bool=True, show:bool=False):
        
        if mixed_precision:
            scaler = GradScaler(self.device, enabled=mixed_precision)

        if is_train: model.train()
        else: model.eval()

        for epoch in range(epochs):
            has_acc = hasattr(model, 'get_accuracy')
            running_loss = 0.0
            running_acc = 0.0
            for i, (x,y) in enumerate(loader):
                x.to(self.device);y.to(self.device)

                optimizer.zero_grad()

                if mixed_precision:
                    with autocast(self.device, enabled=mixed_precision):
                        yhat = model(x)
                        loss = criterion(yhat, y)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                if has_acc: running_acc += model.get_accuracy(yhat, y)

            running_loss += loss.item()
            exp_acc = running_acc / len(loader)
            exp_loss = running_loss / len(loader)
            if scheduler: scheduler.step()
            if (epoch % 5 == 0 or exp_acc >= .99) and show:
                msg = f'Epoch [{epoch + 1}/{epochs}], Train Loss: {exp_loss:.2f}'
                if has_acc: msg+=f', Accuracy: {exp_acc:.2f}'
                print(msg)
                #if exp_acc >= .99: break
        if show: print('Experiment Complete')
        return model
    

    def run_experiment(self,  model:nn.Module, training_dataset:Dataset,
                              testing_dataset:Dataset=None, epochs:int=10,
                              learning_rate:float=0.01, weight_decay=0.01,
                              batch_size:int=100, criterion=None, sampler=None,
                              mixed_precision=None, scheduler=False, pruning=False,
                              show=False, train_shuffle = True, test_shuffle = False):
        # Runs the experiment with the given criteria, returning a trained model

        model = self.model_type() if not model else model
        model.to(self.device)

        if testing_dataset: self.generate_dataloaders_from_datasets(training_dataset, testing_dataset, batch_size, sampler, train_shuffle, test_shuffle)
        else: self.generate_dataloaders_from_full_set(training_dataset, batch_size, sampler, train_shuffle, test_shuffle)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler: scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        self.criterion = criterion()
        
        model = self.epoch_loop(epochs, model, optimizer, self.criterion, self.training_loader, mixed_precision, scheduler, True, show)

        if pruning: model = self.prune_model(model, amount=0.1)
        return model
    
    def evaluate_model(self, model:nn.Module = None, loader = None, show:bool = True):
        # Evaluates the trained model
        if not loader: loader = self.testing_loader
        if not model: model = self.model
        model.to(self.device)

        has_acc = hasattr(model, 'get_accuracy')
        criterion = self.criterion

        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        for i, (x,y) in enumerate(loader):
            x.to(self.device);y.to(self.device)
            yhat = model(x)
            loss = criterion(yhat, y)
            running_loss += loss.item()
            if has_acc: running_acc += model.get_accuracy(yhat, y)
        msg = f'[Evaluation over {len(loader)} Batches], Test Loss: {running_loss/len(loader):.2f}'; self.model_loss = running_loss/len(loader)
        if has_acc: msg+=f', Accuracy: {running_acc/len(loader):.2f}'; self.model_accuracy = running_acc/len(loader)
        if show: print(msg)
        return model
    
    def prune_model(self, model, amount=0.3):
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                return prune.l1_unstructured(module, name='weight', amount=amount)

    def generate_dataloaders_from_full_set(self, dataset:Dataset, batch_size:int, sampler=None, train_shuffle:bool = True, test_shuffle:bool = False):
        '''
            If we only input one dataset that means that we intend to do the standard 80/20 split
        '''
        num_items = len(dataset)
        num_train = round(num_items * 0.8)
        num_val = num_items - num_train
        
        self.training_dataset, self.testing_dataset = random_split(dataset, [num_train, num_val])
        self.training_loader = DataLoader(self.training_dataset,  batch_size=batch_size, sampler=sampler, shuffle=train_shuffle)
        self.testing_loader = DataLoader(self.testing_dataset, batch_size=batch_size, shuffle=test_shuffle)

    def generate_dataloaders_from_datasets(self, training_dataset:Dataset, testing_dataset:Dataset, batch_size:int, sampler=None, train_shuffle:bool = True, test_shuffle:bool = False):
        self.training_dataset, self.testing_dataset = training_dataset, testing_dataset
        self.training_loader = DataLoader(self.training_dataset,  batch_size=batch_size, sampler=sampler, shuffle=train_shuffle)
        self.testing_loader = DataLoader(self.testing_dataset, batch_size=batch_size, shuffle=test_shuffle)