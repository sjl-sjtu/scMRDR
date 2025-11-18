import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

class EarlyStopping:
    '''
    Early stopping for training.
    Args:
        patience: int, patience for early stopping
        delta: float, delta for early stopping
        verbose: bool, whether to print early stopping information
    '''
    def __init__(self, patience=10, delta=0.0, verbose=False):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(model)
            self.counter = 0

    # def save_checkpoint(self, model, path="best_model.pt"):
    #     self.path = path
        # torch.save(model.state_dict(), self.path)
        # if self.verbose:
        #     print(f"Validation loss decreased, model saved to {self.path}")

def train_model(device, writer, train_dataset, validate_dataset, model, epoch_num, batch_size, 
                num_batch, lr, accumulation_steps=1, num_warmup = 0, adaptlr = False, early_stopping=True, patience=25,
                sample_weights=None): #inferenceRNA, inferenceATAC, 
    '''
    Train the model.
    Args:
        device: device to train the model
        writer: writer to write the training progress
        train_dataset: train dataset
        validate_dataset: validate dataset
        model: model to train
        epoch_num: number of epochs
        batch_size: batch size
        num_batch: number of batches
        lr: learning rate
        accumulation_steps: number of steps to accumulate gradients
        num_warmup: number of warmup epochs
        adaptlr: whether to adapt learning rate
        early_stopping: whether to use early stopping
        patience: patience for early stopping
        sample_weights: sample weights for weighted sampling
    '''
    # load data
    if sample_weights is not None:
        sample_weights = torch.tensor(sample_weights,dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_data = DataLoader(train_dataset,batch_size,shuffle=False,sampler=sampler,drop_last=True,num_workers=4,pin_memory=True)
    else:   
        train_data = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_vae = torch.optim.Adam(list(model.encoder_shared.parameters()) + 
                                     list(model.encoder_specific.parameters()) + 
                                     list(model.decoder.parameters()) +
                                     list(model.prior_net_specific.parameters()), lr=lr)
    optimizer_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    if adaptlr==True:
        scheduler_d =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_d,
                                                            T_max =  epoch_num * num_batch)
        scheduler_vae =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer_vae,
                                                            T_max =  epoch_num * num_batch)
    
    if early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=True)   

    for epoch in range(epoch_num):
        model.train()
        total_loss,recon_loss,kl_z,preserve_loss,adv_loss,total_discri_loss = \
            0,0,0,0,0,0
        for step, (X,b,m,i,w) in enumerate(train_data):
            X,b,m,i,w = X.to(device),b.to(device),m.to(device), i.to(device),w.to(device)
            X.requires_grad = True
            b.requires_grad = True
            m.requires_grad = True
            
            # torch.autograd.set_detect_anomaly(True)
            # with torch.autograd.detect_anomaly():
            #     loss.backward() 

            if epoch < num_warmup:
                model.train()
                _, _, loss, loss_dict = model(X,b,m,i,w,stage="warmup")
                # with torch.autograd.detect_anomaly():
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
                if (step + 1) % accumulation_steps == 0:
                    optimizer_vae.step()
                    optimizer_vae.zero_grad()      
                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr_vae/train",scheduler_vae.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:    
                    scheduler_vae.step()
                
                total_loss+=loss.item()
                recon_loss+=loss_dict['recon_loss']
                kl_z+=loss_dict['kl_z']
                preserve_loss+=loss_dict['preserve_loss']
                
                # print(loss)
                if writer is not None:
                    writer.add_scalar("Loss/train", loss.item(), epoch*num_batch+step+1)
                    writer.add_scalar("recon_Loss/train", loss_dict['recon_loss'], epoch*num_batch+step+1)
                    writer.add_scalar("KLz_Loss/train", loss_dict['kl_z'], epoch*num_batch+step+1)
                    writer.add_scalar("preserve_Loss/train", loss_dict['preserve_loss'], epoch*num_batch+step+1)

            elif epoch >= num_warmup:
                ### === Phase A: Train Discriminator === ###
                model.eval()
                model.discriminator.train()
                discri_loss = model(X,b,m,i,w, stage="discriminator")
                # with torch.autograd.detect_anomaly():
                discri_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
                if (step + 1) % accumulation_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()  
                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr_d/train",scheduler_d.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:    
                    scheduler_d.step()

                ### === Phase B: Train VAE, fool Discriminator === ###
                model.train()
                model.discriminator.eval()
                _, _, loss, loss_dict = model(X,b,m,i,w,stage="vae")
                # with torch.autograd.detect_anomaly():
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
                if (step + 1) % accumulation_steps == 0:
                    optimizer_vae.step()
                    optimizer_vae.zero_grad()      
                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr_vae/train",scheduler_vae.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:    
                    scheduler_vae.step()
                
                total_loss+=loss.item()
                recon_loss+=loss_dict['recon_loss']
                kl_z+=loss_dict['kl_z']
                preserve_loss+=loss_dict['preserve_loss']
                adv_loss+=loss_dict['adv_loss']
                total_discri_loss+=discri_loss.item()
                
                # print(loss)
                if writer is not None:
                    writer.add_scalar("Loss/train", loss.item(), epoch*num_batch+step+1)
                    writer.add_scalar("recon_Loss/train", loss_dict['recon_loss'], epoch*num_batch+step+1)
                    writer.add_scalar("KLz_Loss/train", loss_dict['kl_z'], epoch*num_batch+step+1)
                    writer.add_scalar("preserve_Loss/train", loss_dict['preserve_loss'], epoch*num_batch+step+1)
                    writer.add_scalar("adv_Loss/train", loss_dict['adv_loss'], epoch*num_batch+step+1)
                    writer.add_scalar("discri_Loss/train", discri_loss.item(), epoch*num_batch+step+1)
            
        if writer is not None:    
            writer.add_scalar("Loss_epoch/train", total_loss / num_batch, epoch+1)
            writer.add_scalar("recon_Loss_epoch/train", recon_loss / num_batch, epoch+1)
            writer.add_scalar("KLz_Loss_epoch/train", kl_z / num_batch, epoch+1)
            writer.add_scalar("preserve_Loss_epoch/train", preserve_loss / num_batch, epoch+1)
            writer.add_scalar("adv_Loss_epoch/train", adv_loss / num_batch, epoch+1)
            writer.add_scalar("discri_Loss_epoch/train", total_discri_loss / num_batch, epoch+1)
        
        if (epoch + 1) % 1 == 0:
            print("epoch {}: loss = {:.4f}, Recon_loss = {:.4f}, KL_loss = {:.4f}, preserve_loss = {:.4f}, adv_loss = {:.4f}, discri_loss = {:.4f}".format( # 
                epoch+1,total_loss / num_batch, recon_loss / num_batch, kl_z / num_batch, preserve_loss / num_batch, adv_loss / num_batch, total_discri_loss / num_batch)) #
        
        if epoch >= num_warmup:
            if early_stopping:
                validate_loss = validate_model(device, validate_dataset, model, batch_size)
                early_stopping(validate_loss, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

def validate_model(device, validate_dataset, model, batch_size):
    '''
    Validate the model.
    Args:
        device: device to validate the model
        validate_dataset: validate dataset
        model: model to validate
        batch_size: batch size
    '''
    model.eval()
    validate_data = DataLoader(validate_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    total_loss= 0
    for _, (X,b,m,i,w) in enumerate(validate_data):
        X,b,m,i,w = X.to(device),b.to(device),m.to(device), i.to(device),w.to(device)
        _,_,loss,_ = model(X,b,m,i,w,stage="vae")
        total_loss+=loss.item()    
    return loss


def inference_model(device, inference_dataset, model, batch_size):
    '''
    Inference the model.
    Args:
        device: device to inference the model
        inference_dataset: inference dataset
        model: model to inference
        batch_size: batch size
    '''
    model.eval()
    inference_data = DataLoader(inference_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    z1_list = []
    z2_list = []
    total_loss,recon_loss,kl_z,preserve_loss,adv_loss= \
            0,0,0,0,0
    for step, (X,b,m,i,w) in enumerate(inference_data):
        X,b,m,i,w = X.to(device),b.to(device),m.to(device), i.to(device),w.to(device)
        z1,z2,loss,loss_dict = model(X,b,m,i,w,stage="vae")
        z1_list.append(z1.detach().cpu().numpy())
        z2_list.append(z2.detach().cpu().numpy())
        total_loss+=loss.item()
        recon_loss+=loss_dict['recon_loss']
        kl_z+=loss_dict['kl_z']
        preserve_loss+=loss_dict['preserve_loss']
        adv_loss+=loss_dict['adv_loss']
        # total_discri_loss+=discri_loss.item()
        
    z_shared = np.concatenate(z1_list, axis=0)
    z_specific = np.concatenate(z2_list, axis=0)
    num_batch = np.ceil(len(inference_dataset)/batch_size)
    print("inference: loss = {:.4f}, Recon_loss = {:.4f}, KL_loss = {:.4f}, preserve_loss = {:.4f}, adv_loss = {:.4f}".format( # 
          total_loss / num_batch, recon_loss / num_batch, kl_z / num_batch, preserve_loss / num_batch, adv_loss / num_batch))  #
    
    return z_shared, z_specific

