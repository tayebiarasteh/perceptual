"""
Created on Feb 5, 2024.
PathologyAnonym_Train_Valid.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15



class Training:
    def __init__(self, cfg_path, resume=False, label_names=None):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        resume: bool
            if we are resuming training from a checkpoint
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        # self.label_names = label_names
        self.label_names = ['control', 'dysarthria', 'dysglossia', 'dysphonia', 'CLP']

        if resume == False:
            self.model_info = self.params['Network']
            self.epoch = 0
            self.best_loss = float('inf')
            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))
            self.best_metric = 0


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, optimiser, loss_function, weight=None):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)

        self.loss_weight = weight.to(self.device)
        self.loss_function = loss_function(pos_weight=self.loss_weight)
        # self.loss_function = loss_function()
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model, optimiser, loss_function, weight):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                self.params['checkpoint_name']))
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.loss_weight = weight
        self.loss_weight = self.loss_weight.to(self.device)
        self.loss_function = loss_function(weight=self.loss_weight)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)


    def train_epoch(self, train_loader, valid_loader=None, num_epochs=1000):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(num_epochs - self.epoch):
            self.epoch += 1

            # initializing the loss list
            batch_loss = 0
            start_time = time.time()

            for idx, (image, label) in enumerate(train_loader):
                self.model.train()

                image = image.squeeze(0)
                label = label.squeeze(0)

                image = image.to(self.device)
                label = label.to(self.device)

                image = image.float()

                self.optimiser.zero_grad()

                with torch.set_grad_enabled(True):

                    output = self.model(image)
                    # print(image.shape)
                    # print(label.shape)
                    loss = self.loss_function(output, label) # for multilabel

                    loss.backward()
                    self.optimiser.step()

                    batch_loss += loss.item()

            train_loss = batch_loss / len(train_loader)
            self.writer.add_scalar('Train_loss_avg', train_loss, self.epoch)

            # Saves information about training to config file
            self.params['Network']['num_epoch'] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ######## Save a checkpoint every epoch ########
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(),
                        'model_info': self.model_info, 'best_loss': self.best_loss},
                       os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))
            ######## Save a checkpoint every epoch ########

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_F1, valid_AUC, valid_accuracy, valid_specificity, valid_sensitivity, valid_precision, optimal_threshold = self.valid_epoch(valid_loader)
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_F1=valid_F1, valid_AUC=valid_AUC, valid_accuracy=valid_accuracy, valid_specificity=valid_specificity,
                                            valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time, valid_loss=valid_loss, valid_F1=valid_F1,
                                        valid_AUC=valid_AUC, valid_accuracy=valid_accuracy, valid_specificity= valid_specificity,
                                        valid_sensitivity=valid_sensitivity, valid_precision=valid_precision, optimal_thresholds=optimal_threshold)
                else:
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time)




    def train_epoch_multiclass(self, train_loader, valid_loader=None, num_epochs=1000):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for epoch in range(num_epochs - self.epoch):
            self.epoch += 1

            # initializing the loss list
            batch_loss = 0
            start_time = time.time()

            for idx, (image, label) in enumerate(train_loader):
                self.model.train()

                image = image.squeeze(0)
                label = label.squeeze(0)

                image = image.to(self.device)
                label = label.to(self.device)

                image = image.float()

                self.optimiser.zero_grad()

                with torch.set_grad_enabled(True):

                    output = self.model(image)
                    loss = self.loss_function(output, label) # for multilabel

                    loss.backward()
                    self.optimiser.step()

                    batch_loss += loss.item()

            train_loss = batch_loss / len(train_loader)
            self.writer.add_scalar('Train_loss_avg', train_loss, self.epoch)

            # Saves information about training to config file
            self.params['Network']['num_epoch'] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ######## Save a checkpoint every epoch ########
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(),
                        'model_info': self.model_info, 'best_loss': self.best_loss},
                       os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))
            ######## Save a checkpoint every epoch ########

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    valid_loss, valid_F1, valid_AUC, valid_accuracy, valid_specificity, valid_sensitivity, valid_precision, optimal_threshold = self.valid_epoch_multiclass(valid_loader)
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_loss=valid_loss, valid_F1=valid_F1, valid_AUC=valid_AUC, valid_accuracy=valid_accuracy, valid_specificity=valid_specificity,
                                            valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                    self.savings_prints_multiclass(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time, valid_loss=valid_loss, valid_F1=valid_F1,
                                        valid_AUC=valid_AUC, valid_accuracy=valid_accuracy, valid_specificity= valid_specificity,
                                        valid_sensitivity=valid_sensitivity, valid_precision=valid_precision, optimal_thresholds=optimal_threshold)
                else:
                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                    self.savings_prints_multiclass(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_loss, total_time)



    def valid_epoch(self, valid_loader):
        """Validation epoch

        Returns
        -------
        """
        self.model.eval()
        total_f1_score = []
        total_AUROC = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        logits_for_loss_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(valid_loader):

            image = image.squeeze(0)
            label = label.squeeze(0)
            image = image.to(self.device)
            label = label.to(self.device)
            image = image.float()

            with torch.no_grad():
                output = self.model(image)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                logits_for_loss_cache = torch.cat((logits_for_loss_cache, output))
                labels_cache = torch.cat((labels_cache, label))

        ############ Evaluation metric calculation ########

        loss = self.loss_function(logits_for_loss_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        # threshold finding for metrics calculation
        preds_with_sigmoid_cache = preds_with_sigmoid_cache.cpu().numpy()
        labels_cache = labels_cache.int().cpu().numpy()
        optimal_threshold = np.zeros(labels_cache.shape[1])

        for idx in range(labels_cache.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(labels_cache[:, idx], preds_with_sigmoid_cache[:, idx], pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold[idx] = thresholds[optimal_idx]

        predicted_labels = (preds_with_sigmoid_cache > optimal_threshold).astype(np.int32)

        confusion = metrics.multilabel_confusion_matrix(labels_cache, predicted_labels)

        F1_disease = []
        accuracy_disease = []
        specificity_disease = []
        sensitivity_disease = []
        precision_disease = []

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            specificity_disease.append(TN / (TN + FP + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            precision_disease.append(TP / (TP + FP + epsilon))

        # Macro averaging
        total_f1_score.append(np.stack(F1_disease))
        try:
            total_AUROC.append(metrics.roc_auc_score(labels_cache, preds_with_sigmoid_cache, average=None))
        except:
            print('hi')
            pass
        total_accuracy.append(np.stack(accuracy_disease))
        total_specificity_score.append(np.stack(specificity_disease))
        total_sensitivity_score.append(np.stack(sensitivity_disease))
        total_precision_score.append(np.stack(precision_disease))

        # average_loss = total_loss / len(valid_loader)
        average_f1_score = np.stack(total_f1_score).mean(0)
        average_AUROC = np.stack(total_AUROC).mean(0)
        average_accuracy = np.stack(total_accuracy).mean(0)
        average_specificity = np.stack(total_specificity_score).mean(0)
        average_sensitivity = np.stack(total_sensitivity_score).mean(0)
        average_precision = np.stack(total_precision_score).mean(0)

        return epoch_loss, average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision, optimal_threshold



    def valid_epoch_multiclass(self, valid_loader):
        """Validation epoch

        Returns
        -------
        """
        self.model.eval()
        total_f1_score = []
        total_AUROC = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        logits_for_loss_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(valid_loader):

            image = image.squeeze(0)
            label = label.squeeze(0)
            image = image.to(self.device)
            label = label.to(self.device)
            label = label.float()

            with torch.no_grad():
                output = self.model(image)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                logits_for_loss_cache = torch.cat((logits_for_loss_cache, output))
                labels_cache = torch.cat((labels_cache, label))

        ############ Evaluation metric calculation ########

        loss = self.loss_function(logits_for_loss_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        # threshold finding for metrics calculation
        preds_with_sigmoid_cache = preds_with_sigmoid_cache.cpu().numpy()
        labels_cache = labels_cache.int().cpu().numpy()
        optimal_threshold = np.zeros(labels_cache.shape[1])

        for idx in range(labels_cache.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(labels_cache[:, idx], preds_with_sigmoid_cache[:, idx], pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold[idx] = thresholds[optimal_idx]

        predicted_labels = (preds_with_sigmoid_cache > optimal_threshold).astype(np.int32)

        # Metrics calculation (macro) over the whole set
        confusion = metrics.multilabel_confusion_matrix(labels_cache, predicted_labels)

        F1_disease = []
        accuracy_disease = []
        specificity_disease = []
        sensitivity_disease = []
        precision_disease = []

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            specificity_disease.append(TN / (TN + FP + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            precision_disease.append(TP / (TP + FP + epsilon))

        # Macro averaging
        total_f1_score.append(np.stack(F1_disease))
        try:
            total_AUROC.append(metrics.roc_auc_score(labels_cache, preds_with_sigmoid_cache, average=None))
        except:
            print('hi')
            pass
        total_accuracy.append(np.stack(accuracy_disease))
        total_specificity_score.append(np.stack(specificity_disease))
        total_sensitivity_score.append(np.stack(sensitivity_disease))
        total_precision_score.append(np.stack(precision_disease))

        average_f1_score = np.stack(total_f1_score).mean(0)
        average_AUROC = np.stack(total_AUROC).mean(0)
        average_accuracy = np.stack(total_accuracy).mean(0)
        average_specificity = np.stack(total_specificity_score).mean(0)
        average_sensitivity = np.stack(total_sensitivity_score).mean(0)
        average_precision = np.stack(total_precision_score).mean(0)

        return epoch_loss, average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision, optimal_threshold



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, total_time, total_overhead_time=0, total_datacopy_time=0, valid_loss=None, valid_F1=None, valid_AUC=None, valid_accuracy=None,
                       valid_specificity=None, valid_sensitivity=None, valid_precision=None, optimal_thresholds=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specificity: float
            validation specificity of the model

        valid_loss: float
            validation loss of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)
        new_metric = valid_AUC.mean() + valid_accuracy.mean() + valid_F1.mean()

        netto_time = total_time - total_overhead_time - total_datacopy_time
        netto_hours, netto_mins, netto_secs = self.time_duration(0, netto_time)

        # Saving the model based on the best metric
        if new_metric > self.best_metric:
            self.best_metric = new_metric
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                             self.params['network_output_path'], self.params['trained_model_name']))

        # Save a checkpoint every epoch
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss_state_dict': self.loss_function.state_dict(),
                    'model_info': self.model_info, 'best_loss': self.best_loss},
                   os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'epoch: {self.epoch} | '
              f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.1f}s | '
              f'total time: {total_hours}h {total_mins}m {total_secs:.1f}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | avg AUROC: {valid_AUC.mean() * 100:.2f}% | avg accuracy: {valid_accuracy.mean() * 100:.2f}%'
            f' | avg F1: {valid_F1.mean() * 100:.2f}% | avg specificity: {valid_specificity.mean() * 100:.2f}%'
            f' | avg recall (sensitivity): {valid_sensitivity.mean() * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'\n\n----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.1f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.1f}s | ' \
                  f'\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | avg AUROC: {valid_AUC.mean() * 100:.2f}% | avg accuracy: {valid_accuracy.mean() * 100:.2f}% ' \
                   f' | avg F1: {valid_F1.mean() * 100:.2f}% | avg specificity: {valid_specificity.mean() * 100:.2f}%' \
                   f' | avg recall (sensitivity): {valid_sensitivity.mean() * 100:.2f}% | avg precision: {valid_precision.mean() * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}' \
                  f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def savings_prints_multiclass(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, total_time, total_overhead_time=0, total_datacopy_time=0, valid_loss=None, valid_F1=None, valid_AUC=None, valid_accuracy=None,
                       valid_specificity=None, valid_sensitivity=None, valid_precision=None, optimal_thresholds=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specificity: float
            validation specificity of the model

        valid_loss: float
            validation loss of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)
        new_metric = valid_AUC.mean() + valid_accuracy.mean() + valid_F1.mean()

        netto_time = total_time - total_overhead_time - total_datacopy_time
        netto_hours, netto_mins, netto_secs = self.time_duration(0, netto_time)

        # Saving the model based on the best metric
        if new_metric > self.best_metric:
            self.best_metric = new_metric
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                             self.params['network_output_path'], self.params['trained_model_name']))

        # Save a checkpoint every epoch
        torch.save({'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'loss_state_dict': self.loss_function.state_dict(),
                    'model_info': self.model_info, 'best_loss': self.best_loss},
                   os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'epoch: {self.epoch} | '
              f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.1f}s | '
              f'total time: {total_hours}h {total_mins}m {total_secs:.1f}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | avg AUROC: {valid_AUC.mean() * 100:.2f}% | avg accuracy: {valid_accuracy.mean() * 100:.2f}%'
            f' | avg F1: {valid_F1.mean() * 100:.2f}% | avg specificity: {valid_specificity.mean() * 100:.2f}%'
            f' | sensitivity: {valid_sensitivity.mean() * 100:.2f}%\n')

            # saving the training and validation stats
            msg = f'\n\n----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.1f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.1f}s | ' \
                  f'\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | avg AUROC: {valid_AUC.mean() * 100:.2f}% | avg accuracy: {valid_accuracy.mean() * 100:.2f}% ' \
                   f' | avg F1: {valid_F1.mean() * 100:.2f}% | avg specificity: {valid_specificity.mean() * 100:.2f}%' \
                   f' | sensitivity: {valid_sensitivity.mean() * 100:.2f}% | avg precision: {valid_precision.mean() * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'epoch: {self.epoch} | epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                   f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s\n\n\ttrain loss: {train_loss:.4f}' \
                  f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)


        if valid_loss:
            msg = f'Individual AUROC:\n'
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                f.write(msg)
            for idx, pathology in enumerate(self.label_names):
                try:
                    msg = f'{pathology}: {valid_AUC[idx] * 100:.2f} | '
                except:
                    msg = f'{pathology}: {valid_AUC * 100:.2f} | '

                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                    f.write(msg)

            msg = f'\n\nIndividual accuracy:\n'
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                f.write(msg)
            for idx, pathology in enumerate(self.label_names):
                msg = f'{pathology}: {valid_accuracy[idx] * 100:.2f} | '
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                    f.write(msg)

            msg = f'\n\nIndividual specificity scores:\n'
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                f.write(msg)
            for idx, pathology in enumerate(self.label_names):
                msg = f'{pathology}: {valid_specificity[idx] * 100:.2f} | '
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                    f.write(msg)

            msg = f'\n\nIndividual sensitivity scores:\n'
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                f.write(msg)
            for idx, pathology in enumerate(self.label_names):
                msg = f'{pathology}: {valid_sensitivity[idx] * 100:.2f} | '
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
                    f.write(msg)




    def calculate_tb_stats(self, valid_loss=None, valid_F1=None, valid_AUC=None, valid_accuracy=None, valid_specificity=None, valid_sensitivity=None, valid_precision=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specificity: float
            validation specificity of the model

        valid_loss: float
            validation loss of the model
        """
        if valid_loss is not None:
            self.writer.add_scalar('Valid_loss', valid_loss, self.epoch)
            self.writer.add_scalar('valid_avg_F1', valid_F1.mean(), self.epoch)
            self.writer.add_scalar('Valid_avg_AUROC', valid_AUC.mean(), self.epoch)

            self.writer.add_scalar('Valid_avg_accuracy', valid_accuracy.mean(), self.epoch)