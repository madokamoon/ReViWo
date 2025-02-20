import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    import deepspeed
except Exception as e:
    print(f"Failed to import deepspeed: {e}")
    pass
from collections import deque
from tqdm import tqdm
import numpy as np
import os
project_dir = str(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from common.loggers import Logger

def get_lr_lambda(initial_lr, final_lr, num_epochs):
    def lr_lambda(epoch):
        if epoch <= num_epochs:
            return (final_lr / initial_lr) ** (epoch / num_epochs)
        else:
            return final_lr / initial_lr
    return lr_lambda

    
class BaseTrainer():
    def __init__(self, config, logger: Logger, use_deepspeed=True):
        self.config = config
        self.use_deepspeed = use_deepspeed
        self.logger = logger

        model = self.init_model(config)
        if config.load_ckpt_path != "none":
            model.load_checkpoint(config.load_ckpt_path)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.optimizer.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(config.optimizer.lr, config.optimizer.final_lr, config.optimizer.lr_num_step))

        self.dataset = self.init_dataset(config)

        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.initialize(model, optimizer, self.dataset, lr_scheduler, config)

    @property
    def algo_name(self):
        return "Algorithm"

    @property
    def device(self):
        if self.use_deepspeed:
            return self.model.local_rank
        else:
            return "cuda"
        
    @property
    def save_checkpoint(self):
        if self.use_deepspeed:
            return self.model.save_checkpoint
        else:
            return self.model.custom_save_checkpoint
        
    @property
    def load_checkpoint(self):
        if self.use_deepspeed:
            return self.model.load_checkpoint
        else:
            return self.model.custom_load_checkpoint

    def step_loss(self, loss: torch.Tensor):
        if self.use_deepspeed:
            self.model.backward(loss)
            self.model.step()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def init_model(self, config) -> nn.Module:
        raise NotImplementedError("Please implement the init_model method in the subclass")

    def init_dataset(self, config) -> Dataset:
        raise NotImplementedError("Please implement the init_dataset method in the subclass")
    
    def get_loss(self, *args, **kwargs):
        raise NotImplementedError("Please implement the get_loss method in the subclass")

    def eval(self, *args, **kwargs):
        raise NotImplementedError("Please implement the eval method in the subclass")
    

    def initialize(self, model, optimizer, dataset, lr_scheduler, config):
        if self.use_deepspeed:
            return deepspeed.initialize(
            config=dict(config.deepspeed),
            model=model,
            optimizer=optimizer,
            training_data=dataset, 
            lr_scheduler=lr_scheduler,
        )
        else:
            model = model.to(self.device)
            dataloader = DataLoader(dataset, config.trainer.batch_size, shuffle=True)
            return model, optimizer, dataloader, lr_scheduler
 
    def train(self, ):
        losses = deque(maxlen=100)
        min_eval_error = 10000
        timesteps = 0

        for epoch in range(self.config.trainer.num_epochs):
            for _, batch in tqdm(enumerate(self.dataloader), desc=f"Epoch {epoch}, timestep {timesteps+1}"):
                self.model.train()
                timesteps += 1
                
                loss_dict = self.model(batch)
                loss = self.get_loss(loss_dict)
                self.step_loss(loss)
                losses.append(loss.item())

                if timesteps % self.config.trainer.log_step == 0:
                    print(f"Training Loss: {np.mean(losses)}")

                    self.logger.set_timestep(timesteps)
                    self.logger.logkv(f"{self.algo_name}/Train loss", np.mean(losses))
                    
                    if timesteps % self.config.trainer.eval_step == 0:
                        eval_error = self.eval()
                        self.logger.logkv(f"{self.algo_name}/Eval error", eval_error)
                        if eval_error < min_eval_error:
                            min_eval_error = eval_error
                            self.logger.save_model(self.save_checkpoint, 1, save_dir="best_model")

                    self.logger.dumpkvs()
                
                if timesteps % self.config.trainer.save_step == 0:
                    self.logger.save_model(self.save_checkpoint, self.config.trainer.max_ckpt, save_dir=f"model/step_{timesteps}")

            if not self.use_deepspeed:
                self.lr_scheduler.step(timesteps)

            

    
    

        