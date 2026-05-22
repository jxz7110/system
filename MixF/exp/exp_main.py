from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from model import MixF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, attention_map
from utils.metrics import metric, MAE, MSE
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb
import random
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
warnings.filterwarnings('ignore')

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        l1_loss = self.l1_loss(outputs, targets)
        l2_loss = self.l2_loss(outputs, targets)

        loss = self.alpha * l1_loss + self.beta * l2_loss
        return loss


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'MixF': MixF
        }
            
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss_flag==1: # loss_flag 0 for MSE, 1 for MAE, 2 for both of MSE & MAE, 3 for SmoothL1loss
            criterion = nn.L1Loss()
        elif self.args.loss_flag==2:
            criterion = MultiTaskLoss(alpha=0.5, beta=0.5)
        elif self.args.loss_flag == 3:
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                # print("vali_data:",batch_x.shape,batch_y.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                preds.append(pred.numpy())
                trues.append(true.numpy())
                total_loss.append(loss)
        preds = np.array(preds)
        trues = np.array(trues)
        total_loss = np.average(total_loss)
        print("vali loss: {}".format(MSE(preds, trues)))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        epoch_time_sum = 0
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            if self.args.wandb_if:
                wandb.log({
                    "Epoch": epoch+1,
                    "Train Loss": train_loss,
                    "Valid loss": vali_loss,
                    "Test loss": test_loss
                })
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                # result save
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                f = open("result.txt", 'a')
                f.write('epoch_time_sum:{}, epoch:{}, epoch_time_avg:{}'.format(epoch_time_sum, epoch + 1,
                                                                                   epoch_time_sum / (epoch + 1)))
                f.write('\n')
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        print(test_data)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        infer_time_sum = 0
        batch_sum = len(test_data)
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + self.args.model_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + 'link.pdf'), self.args.seq_len)
                    attention_map(self.model, name=os.path.join(folder_path, str(i) + 'hot.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model,(batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        print(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './test_results/' + self.args.model_id + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # for name, param in self.model.named_parameters():
        #     print(f"Name: {name}, Shape: {param.shape}")

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.root_path,'MixF',self.args.checkpoints)
            best_model_path = path + 'checkpoint.pth'
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # print(preds)

        # result save
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        #这部分保存预测结果
        ##这部分的想法是和之前的excel文件结合起来  将输出的结果进行拼接
        # 写一个具体的函数实现保存吧！
        old_path=os.path.join(self.args.root_path,self.args.data_path)
        self.save_pred2excel(old_path=old_path, new_path="/mnt/data2/jingxz/system/data_message/predicts", preds=preds)
        # np.save(folder_path + 'real_prediction.npy', preds)

        return
    def save_pred2excel(self, old_path, new_path, preds):
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        df_raw = pd.read_excel(old_path)
        cols = list(df_raw.columns)
        # print(cols)
        lasttime = df_raw['monthes'].iloc[-1]
        lasttime = datetime.strptime(lasttime, "%Y-%m")
        newtimes=[]
        preds = np.squeeze(preds)
        # print(preds.shape)
        for i in range(1,len(preds)+1):
            new_date = lasttime + relativedelta(months=i)
            new_date = new_date.strftime('%Y-%m')
            newtimes.append(new_date)
        newtimes=np.array(newtimes)
        # print(newtimes.shape, preds.shape)
        #这里得到含有时间信息的预测值
        preds = np.column_stack((newtimes, preds))
        # print(df_raw.shape,preds.shape)
        preds = pd.DataFrame(preds, columns=cols)
        new_pd = pd.concat([df_raw, preds], axis=0).reset_index(drop=True)
        # print(new_pd)
        if self.args.do_key == False:
            path=new_path+'.xlsx'
        else:
            path=f'{new_path}/{self.args.key}.xlsx'
        print(path)
        with pd.ExcelWriter(path, engine='openpyxl', mode='w') as writer:  # 使用 'w' 模式重写文件
            new_pd.to_excel(writer, sheet_name='predicts', index=False)

