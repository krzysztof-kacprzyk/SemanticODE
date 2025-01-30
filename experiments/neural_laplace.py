import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torchlaplace import laplace_reconstruct



class LaplaceRepresentationFunc(nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        # output_dim = 1
        # latent_dim and hidden_units are hyperparameters
        super(LaplaceRepresentationFunc, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        phi_max = torch.pi / 2.0
        self.phi_scale = phi_max - -torch.pi / 2.0

    def forward(self, i):
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0
            - torch.pi / 2.0
            + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi
    
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=3, hidden_units=64, dropout_rate=0.0):
        super(SimpleEncoder, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_units, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class NeuralLaplaceRegressor(pl.LightningModule):
    def __init__(
        self,
        latent_dim = 2,
        hidden_units = 64,
        num_layers = 3,
        learning_rate=1e-3,
        weight_decay=0.0,
        dropout_rate=0.0,
        device='cpu',
        seed=0,
        batch_size=32,
    ):
        super(NeuralLaplaceRegressor, self).__init__()
        
        # Set seed for reproducibility
        self.seed = seed
        self.batch_size = batch_size
        self._set_seed()



        s_dim = 33

        self.save_hyperparameters() # this will make the arguments accessible through self.hparams
        self.encoder = SimpleEncoder(input_dim=1, 
                                     latent_dim=latent_dim, 
                                     num_layers=num_layers, 
                                     hidden_units=hidden_units // 2, 
                                     dropout_rate=dropout_rate)
        self.laplace_representation_func = LaplaceRepresentationFunc(s_dim=s_dim, 
                                                                     output_dim=1, 
                                                                     latent_dim=latent_dim, 
                                                                     hidden_units=hidden_units)
        self.loss_fn = nn.MSELoss()

        # Data normalization parameters
        self.y_mean = None
        self.y_std = None
        self.t_mean = None
        self.t_std = None



    def _set_seed(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        pl.seed_everything(self.seed, workers=True)

    def forward(self, x0_batch, t):
        # x0_batch: tensor of shape [batch_size, 1]
        # t: tensor of shape [batch_size, sequence_length]

        # Encode the initial conditions
        latent = self.encoder(x0_batch)

        # Compute the Laplace representation
        predictions = laplace_reconstruct(self.laplace_representation_func, latent, t, recon_dim=1)
        
        return predictions.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x0_batch, t_batch, y_true_batch = batch
        pred_y_batch = self.forward(x0_batch, t_batch)
        loss = self.loss_fn(pred_y_batch, y_true_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x0_batch, t_batch, y_true_batch = batch
        pred_y_batch = self.forward(x0_batch, t_batch)
        loss = self.loss_fn(pred_y_batch, y_true_batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),  # Changed from self.func.parameters()
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def _normalize_data(self, xs, ys, ts):

        self.y_mean = ys.mean()
        self.y_std = ys.std()
        self.t_mean = ts.mean()
        self.t_std = ts.std()

    def fit(self, train_xs, train_ts, train_ys, val_xs, val_ts, val_ys, max_epochs=100, tuning=False):
        """
        Fits the Neural ODE model to the provided training data with early stopping on validation data.

        Parameters:
        - train_xs (numpy array): Initial conditions for training trajectories, shape [num_train_samples]
        - train_ts (numpy array): Time points for training trajectories, shape [sequence_length]
        - train_ys (numpy array): Observed training trajectories, shape [num_train_samples, sequence_length]
        - val_xs (numpy array): Initial conditions for validation trajectories, shape [num_val_samples]
        - val_ts (numpy array): Time points for validation trajectories, shape [sequence_length]
        - val_ys (numpy array): Observed validation trajectories, shape [num_val_samples, sequence_length]
        - batch_size (int): Number of samples per batch.
        - max_epochs (int): Number of training epochs.
        """
        # Move model to the correct device
        if self.hparams.device == 'cpu':
            device = torch.device('cpu')
        elif self.hparams.device in ['gpu', 'cuda']:
            device = torch.device('cuda')

        # Convert numpy arrays to torch tensors
        train_xs = torch.tensor(train_xs).to(device)
        train_ts = torch.tensor(train_ts).to(device)
        train_ys = torch.tensor(train_ys).to(device)
        val_xs = torch.tensor(val_xs).to(device)
        val_ts = torch.tensor(val_ts).to(device)
        val_ys = torch.tensor(val_ys).to(device)

        self.to(device)

        # Normalize data
        self._normalize_data(train_xs, train_ys, train_ts)

        # Normalize training data
        train_xs_norm = ((train_xs.view(-1, 1) - self.y_mean) / self.y_std).float()
        train_ts_norm = ((train_ts - self.t_mean) / self.t_std).float() 
        train_ys_norm = ((train_ys - self.y_mean) / self.y_std).float()


        # Create training dataset and loader
        train_dataset = TensorDataset(train_xs_norm, train_ts_norm, train_ys_norm)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Normalize validation data
        val_xs_norm = ((val_xs.view(-1, 1) - self.y_mean) / self.y_std).float()
        val_ts_norm = ((val_ts - self.t_mean) / self.t_std).float()
        val_ys_norm = ((val_ys - self.y_mean) / self.y_std).float()

        # Create validation dataset and loader
        val_dataset = TensorDataset(val_xs_norm, val_ts_norm, val_ys_norm)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Set up early stopping and model checkpointing
        early_stopping = EarlyStopping('val_loss',
                                       patience=10 if tuning else 20,
                                        mode='min')
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='best_model',
            save_top_k=1,
            mode='min'
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
        
        try:
            import wandb
            wandb_logger = WandbLogger(project="SemanticODE", save_dir="lightning_logs")
        except ImportError:
            wandb_logger = None

        # Initialize the Trainer with the specified device
        trainer = pl.Trainer(
            max_epochs = max_epochs if tuning else max_epochs * 5,
            callbacks=[early_stopping, checkpoint_callback],
            logger=wandb_logger,
            enable_checkpointing=True,
            accelerator=accelerator,
            devices=devices,
            deterministic=True,
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            enable_progress_bar=True
        )

        # Fit the model
        trainer.fit(self, train_loader, val_loader)

        # Load the best model
        self.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])

    def predict(self, xs, ts):
        """
        Predicts trajectories based on initial conditions and time points.

        Parameters:
        - xs (numpy array): Initial conditions, shape [num_samples]
        - ts (numpy array): Time points, shape [sequence_length] or [num_samples, sequence_length]

        Returns:
        - preds (numpy array): Predicted trajectories, shape [num_samples, sequence_length]
        """
        # Ensure normalization parameters are set
        if self.y_mean is None or self.y_std is None or self.t_mean is None or self.t_std is None:
            raise ValueError("Normalization parameters are not set. Please call the fit method first.")

        # Move model to the correct device
        if self.hparams.device == 'cpu':
            device = torch.device('cpu')
        elif self.hparams.device in ['gpu', 'cuda']:
            device = torch.device('cuda')

        self.to(device)

        # Convert numpy arrays to torch tensors
        xs_tensor = torch.tensor(xs).to(device)
        ts_tensor = torch.tensor(ts).to(device)

        # Move normalization parameters to the same device as input tensors
        y_mean = self.y_mean.to(device)
        y_std = self.y_std.to(device)
        t_mean = self.t_mean.to(device)
        t_std = self.t_std.to(device)

        xs_tensor = ((xs_tensor.view(-1, 1) - y_mean) / y_std).float()

        self.eval()
        with torch.no_grad():
            ts_norm = ((ts_tensor - t_mean) / t_std).float()
            pred_y = self.forward(xs_tensor, ts_norm)
            pred_y_denorm = pred_y * y_std + y_mean
            return pred_y_denorm.cpu().numpy()
