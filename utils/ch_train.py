import torch
from huggingface_hub.repocard_data import eval_results_to_model_index
from sympy.concrete.summations import eval_sum
from torch import nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.ch_model import rob_hub_mamba
import random
import numpy as np
from utils.data_loader import data_loader
from utils.plot import plot_loss, plot_acc
from torch.cuda.amp import autocast, GradScaler
# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


def count_model_parameters(model, detailed=False):
    total_params = 0
    layer_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            num_params = param.numel()
            total_params += num_params
            if detailed:
                layer_params[name] = num_params
    if detailed:
        print("Detailed Param Counts by Layer:")
        for name, count in layer_params.items():
            print(f"layer: {name} | Params: {count}")
    print(f"total Trainble Params: {total_params}")
    return total_params

class ChConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                 train_mode = 'regression',
                 model_save_path = '',
                 learning_rate = 5e-6,
                 epochs = 20,
                 dataset_name = 'sims',
                 early_stop = 8,
                 seed = 42,
                 dropout=0.1,
                 model='mamba',
                 batch_size = 16,
                 fuse_version = 'v1',
                 num_hidden_layers = 0
                ):

        self.train_mode = train_mode
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.fuse_version = fuse_version
        self.num_hidden_layers = num_hidden_layers
        
        
class ChTrainer():
    def __init__(self, config):

        self.config = config
        print(f"config are as follows: \n"
              f"batch_size: {config.batch_size}, \n"
              f"cme_version: {config.fuse_version}, \n"
              f"dataset_name: {config.dataset_name}, \n"
              f"dropout: {config.dropout}, \n"
              f"early_stop: {config.early_stop}, \n"
              f"learning_rate: {config.learning_rate}, \n"
              f"model: {config.model}, \n"
              f"seed: {config.seed}, \n")

        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)

        self.scaler = GradScaler()
    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)
        total_loss = 0
        # Loop over all batches.         
        for batch in tqdm(data_loader):                    
            text_inputs = batch["text_tokens"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            targets = batch["targets"].view(-1,1).to(device)
            optimizer.zero_grad()                    # To zero out the gradients.
            with autocast():
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)
                loss = self.criterion(outputs, targets)

            # Compute the training loss.

            #loss.backward()
            #optimizer.step()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item() * text_inputs.size(0)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"Peak GPU Memory Usage: {peak_memory:.2f} MB")

        total_loss = round(total_loss / len(data_loader.dataset), 4)
        return total_loss

    def do_test(self, model, data_loader):
        model.eval()                                # Put the model in training mode.              
        y_pred = []
        y_true = []
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):                    # Loop over all batches.
                text_inputs = batch["text_tokens"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                targets = batch["targets"].view(-1,1).to(device)
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)  # Predictions from 1 batch of data.
                # Compute the training loss.
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()*text_inputs.size(0)

                y_pred.append(outputs.cpu())
                y_true.append(targets.cpu())

            total_loss = round(total_loss / len(data_loader.dataset), 4)
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss

        return eval_results

def ChRun(config):
    print(f"config are as follows: \n"
          f"batch_size: {config.batch_size}, \n"
          f"model_version: {config.fuse_version}, \n"
          f"dataset_name: {config.dataset_name}, \n"
          f"dropout: {config.dropout}, \n"
          f"early_stop: {config.early_stop}, \n"
          f"learning_rate: {config.learning_rate}, \n"
          f"model: {config.model}, \n"
          f"seed: {config.seed}, \n")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name)
    model = rob_hub_mamba(config).to(device)

    for param in model.hubert_model.parameters():
        param.requires_grad = False
               
    trainer = ChTrainer(config)
    total_params = count_model_parameters(model, detailed=True)
    print(f"Total Parameters (PARA): {total_params}")
    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0

    if config.dataset_name == 'simi':
        acc2 = 0.73
        acc5 = 0.40
    else:
        acc2 = 0.83
        acc5 = 0.54
    num_model = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        total_los = trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader)

        if eval_results['Loss']<lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), config.model_save_path+'loss.pth')
            best_epoch = epoch
        if eval_results['Mult_acc_2']>=highest_eval_acc:
            highest_eval_acc = eval_results['Mult_acc_2']
            torch.save(model.state_dict(), config.model_save_path+'acc.pth')
        if eval_results['Mult_acc_2'] >= acc2 and eval_results['Mult_acc_5'] >= acc5:
            model_save_path = os.path.join(config.model_save_path, f'acc2&7high{num_model}.pth')
            torch.save(model.state_dict(), model_save_path)
            num_model += 1
        if epoch - best_epoch >= config.early_stop:
            break


    model.load_state_dict(torch.load(config.model_save_path + 'loss.pth'))
    test_results_loss = trainer.do_test(model, test_loader)
    print('%s: >> ' %('TEST (lowest val loss)  ') + dict_to_str(test_results_loss))

    model.load_state_dict(torch.load(config.model_save_path + 'acc.pth'))
    test_results_acc  = trainer.do_test(model, test_loader)
    print('%s: >> ' %('TEST (highest val acc)') + dict_to_str(test_results_acc))

    for index in range(num_model):
        model.load_state_dict(torch.load(config.model_save_path + f'acc2&7high{index}.pth'))
        test_results_loss = trainer.do_test(model, test_loader)
        print('\n%s: >> ' % (f'TEST (highest val acc2&acc7)[{index}] ') + dict_to_str(test_results_loss))
