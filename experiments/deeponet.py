import numpy as np
import torch
import torch.nn as nn
import lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

class DeepONet(pl.LightningModule):
    def __init__(
        self,
        num_hidden=50,
        num_layers=2,
        learning_rate=1e-3,
        batch_size=32,
        device='cpu',
        dropout_rate=0.0,
        weight_decay=0.0,
        seed=0
    ):
        super(DeepONet, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.seed = seed

        self._set_seed()

        # Branch network processes the initial condition x0
        branch_layers = []
        input_dim_branch = 1  # x0 is scalar
        for _ in range(num_layers):
            branch_layers.append(nn.Linear(input_dim_branch, num_hidden))
            branch_layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                branch_layers.append(nn.Dropout(dropout_rate))
            input_dim_branch = num_hidden
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk network processes the time points t
        trunk_layers = []
        input_dim_trunk = 1  # t is scalar
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(input_dim_trunk, num_hidden))
            trunk_layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                trunk_layers.append(nn.Dropout(dropout_rate))
            input_dim_trunk = num_hidden
        self.trunk_net = nn.Sequential(*trunk_layers)

        # Initialize scalers
        self.x_scaler = StandardScaler()
        self.t_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        pl.seed_everything(self.seed, workers=True)

    def forward(self, x0, t):
        # x0: [batch_size, 1]
        # t: [batch_size, N]
        # Process x0 through branch network
        branch_output = self.branch_net(x0)  # [batch_size, num_hidden]

        # Process t through trunk network
        batch_size, N = t.shape
        t_flat = t.view(-1, 1)  # Flatten to [batch_size*N, 1]
        trunk_output = self.trunk_net(t_flat)  # [batch_size*N, num_hidden]

        # Combine outputs
        branch_output_expanded = branch_output.unsqueeze(1).expand(-1, N, -1)
        branch_output_flat = branch_output_expanded.reshape(-1, self.hparams.num_hidden)

        # Element-wise multiplication and summation (dot product)
        output = (branch_output_flat * trunk_output).sum(dim=1)  # [batch_size*N]
        output = output.view(batch_size, N)

        return output

    def training_step(self, batch, batch_idx):
        x0, t, y_true = batch
        y_pred = self.forward(x0, t)
        loss = nn.MSELoss()(y_pred, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, t, y_true = batch
        y_pred = self.forward(x0, t)
        val_loss = nn.MSELoss()(y_pred, y_true)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, X_train, T_train, Y_train, X_val, T_val, Y_val, max_epochs=200, tuning=False):
        # Data scaling
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        T_train_scaled = self.t_scaler.fit_transform(T_train.reshape(-1, 1)).reshape(T_train.shape)
        Y_train_scaled = self.y_scaler.fit_transform(Y_train)

        X_val_scaled = self.x_scaler.transform(X_val)
        T_val_scaled = self.t_scaler.transform(T_val.reshape(-1, 1)).reshape(T_val.shape)
        Y_val_scaled = self.y_scaler.transform(Y_val)

        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(T_train_scaled, dtype=torch.float32),
            torch.tensor(Y_train_scaled, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(T_val_scaled, dtype=torch.float32),
            torch.tensor(Y_val_scaled, dtype=torch.float32),
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Callbacks for early stopping and checkpointing
        early_stopping = EarlyStopping(monitor='val_loss', patience=10 if tuning else 20, mode='min')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='best_model',
            save_top_k=1,
            mode='min',
        )

        # Determine accelerator and devices based on the specified device
        if self.hparams.device == 'cpu':
            accelerator = 'cpu'
            devices = 1
        elif self.hparams.device in ['gpu', 'cuda']:
            accelerator = 'gpu'
            devices = 1  # Adjust this if you want to use multiple GPUs
        else:
            raise ValueError(f"Unsupported device type: {self.hparams.device}")
        
        wandb_logger = WandbLogger(project="SemanticODE",save_dir="lightning_logs")

        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs if tuning else max_epochs*5,
            callbacks=[early_stopping, checkpoint_callback],
            logger=wandb_logger,
            enable_checkpointing=True,
            accelerator=accelerator,
            devices=devices,
            deterministic=True,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            enable_progress_bar=False
        )

        # Fit the model
        trainer.fit(self, train_loader, val_loader)

        # Load best model after training
        best_model_path = checkpoint_callback.best_model_path
        self.load_state_dict(torch.load(best_model_path)["state_dict"])

    def predict(self, X, T):
        self.eval()
        with torch.no_grad():
            X_scaled = self.x_scaler.transform(X)
            T_scaled = self.t_scaler.transform(T.reshape(-1, 1)).reshape(T.shape)
            x0 = torch.tensor(X_scaled, dtype=torch.float32)
            t = torch.tensor(T_scaled, dtype=torch.float32)
            y_pred_scaled = self.forward(x0, t)
            y_pred = self.y_scaler.inverse_transform(y_pred_scaled.numpy())
        return y_pred
