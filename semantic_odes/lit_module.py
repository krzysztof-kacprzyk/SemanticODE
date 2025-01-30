import lightning as L
import torch
import os
from semantic_odes.model_torch import CubicModel

class LitSketchODE(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        L.seed_everything(self.config['seed'])
        self.model = CubicModel(config)
        self.lr = self.config['lr']
        if 'refit' in self.config:
            self.refit = self.config['refit']
        else:
            self.refit = False
      

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, B, T = batch
        return self.model.forward(X,B,T)
    
    def forward(self, X, B, T):
        return self.model.forward(X, B, T)

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()

        def closure():
            X, B, T, Y = batch
            opt.zero_grad()
            loss = self.model.loss(X, B, T, Y, with_derivative_loss=True)
            
            # print(loss)
            self.manual_backward(loss)
            self.log('train_loss', loss)
            return loss

        opt.step(closure=closure)

    def validation_step(self, batch, batch_idx):

        X, B, T, Y = batch
        loss = self.model.loss(X, B, T, Y, with_derivative_loss=False)

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):

        X, B, T, Y = batch
        loss = self.model.loss(X, B, T, Y, with_derivative_loss=False)
        self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.config['weight_decay'])
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr, history_size=100, max_iter=20, line_search_fn='strong_wolfe')
        # params=list()
        # params.extend(list(self.model.parameters()))
        # optimizer = LBFGSNew(self.model.parameters(), lr=self.lr, cost_use_gradient=True, history_size=100, max_iter=20)
        return optimizer