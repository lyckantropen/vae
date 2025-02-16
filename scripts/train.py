import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm

from ktrojano_vae.vae import ImageVae, ImageVaeLoss

logging.basicConfig(level=logging.INFO)
# set up logger for this script based on file name
logger = logging.getLogger(Path(__file__).stem)

DEFAULT_ARGS: Dict[str, Any] = {
    'run_name': None,
    'batch_size': 96,
    'embed_factor': 1.0,
    'beta': 0.001,
    'beta_max': 1.0,
    'beta_annealing_period': 20,
    'likelihood_type': 'bce',
    'hidden_dim': 32,
    'expand_dim_enc': 32,
    'expand_dim_dec': 24,
    'learning_rate': 1e-2,
    'num_epochs': 1000,
    'type_of_embedding': 'cat',
    'model_version': None
}


def set_up_tensorboard(root, run_name):
    """Set up TensorBoard for logging."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(f'{root}/{run_name}')
    except ImportError as e:
        logger.error('Tensorboard not available')
        raise e


def test_epoch(model: ImageVae,
               dataloader: DataLoader,
               criterion: ImageVaeLoss,
               embed_factor: float,
               type_of_embedding: str = 'cat'
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Test the model on the test dataset for one epoch."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            x, targets = batch
            x = x.to(model.device)
            targets = targets.to(model.device)
            embed = embed_labels_into_input(x, targets, embed_factor, type_of_embedding)
            x_hat, z_mean, z_logv = model(embed)
            loss, _, _ = criterion(x, x_hat, z_mean, z_logv, model.sigma_x)
            losses.append(loss)
    return torch.stack(losses).mean(), x_hat[0:1].detach()


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert labels to one-hot encoding."""
    y = torch.eye(num_classes, device=labels.device)
    return y[labels]


def embed_labels_into_input(x: torch.Tensor, targets: torch.Tensor, embed_factor: float, type_of_embedding: str = 'cat') -> torch.Tensor:
    """Embed the targets into the input."""
    if type_of_embedding == 'none':
        embed = x
    elif type_of_embedding == 'cat':
        embed = torch.cat([x, embed_factor*targets.view(-1, 1, 1, 1).repeat(1, 1, x.size(-2), x.size(-1))], dim=1)
    elif type_of_embedding == 'one_hot':
        one_hot_targets = one_hot_embedding(targets, num_classes=10)
        embed = torch.cat([x, embed_factor*one_hot_targets.view(-1, 10, 1, 1).repeat(1, 1, x.size(-2), x.size(-1))], dim=1)
    else:
        raise ValueError(f'Unknown type of embedding: {type_of_embedding}')
    return embed


class VaeTraining:
    """
    A class used to train a Variational Autoencoder (VAE) model on the MNIST dataset.

    Parameters
    ----------
    base_run_name : str
        The base name for the run.
    initial_checkpoint_path : Optional[str], optional
        Path to the initial checkpoint to resume training from, by default None.
    runs_dir : Path, optional
        Directory to save the runs, by default Path('runs').
    resume_if_exists : bool, optional
        Whether to resume training if a checkpoint exists, by default False.
    override_saved_args : bool, optional
        Whether to override saved arguments with new ones, by default False.
    reset_optimizer : Union[bool, str], optional
        Whether to reset the optimizer, by default False.
    data_root : Path, optional
        Root directory for the data, by default Path('data').
    detect_posterior_collapse : bool, optional
        Whether to detect posterior collapse, by default False.
    **args
        Additional arguments for the training configuration.

    Methods
    -------
    _setup_mnist()
        Sets up the MNIST dataset and DataLoaders.
    _load_args_from_dict(args)
        Loads arguments from a dictionary.
    _get_args_dict()
        Returns the current arguments as a dictionary.
    _create_model()
        Creates the VAE model, optimizer, and loss function.
    _get_run_name(base_name)
        Generates a run name based on the base name and current arguments.
    _resume_from_checkpoint(override_saved_args, model_state_only, reset_optimizer, **args)
        Resumes training from a checkpoint.
    _start_from_scratch()
        Starts training from scratch.
    _save_checkpoint(epoch, test_loss, is_best)
        Saves a checkpoint of the current model state.
    _generate_test_samples(epoch)
        Generates test samples and logs them to TensorBoard.
    train()
        The main training loop for the VAE model.
    """

    def __init__(self,
                 base_run_name: str,
                 initial_checkpoint_path: Optional[str] = None,
                 runs_dir: Path = Path('runs'),
                 resume_if_exists: bool = False,
                 override_saved_args: bool = False,
                 reset_optimizer: Union[bool, str] = False,
                 data_root: Path = Path('data'),
                 detect_posterior_collapse: bool = False,
                 **args
                 ) -> None:
        self.model: ImageVae
        self.optimizer: optim.Optimizer
        self.criterion: ImageVaeLoss
        self.dataset: datasets.MNIST
        self.transform: transforms.Compose
        self.dataloader_train: DataLoader
        self.dataloader_test: DataLoader
        self.batch_size: int
        self.embed_factor: float
        self.beta: float
        self.beta_max: float
        self.beta_annealing_period: int
        self.likelihood_type: str
        self.hidden_dim: int
        self.expand_dim_enc: int
        self.expand_dim_dec: int
        self.learning_rate: float
        self.num_epochs: int
        self.type_of_embedding: str
        self.run_name: str
        self.model_version: Optional[str]
        self.scheduler: optim.lr_scheduler.ReduceLROnPlateau

        self.original_run_name: Optional[str] = None
        self.start_epoch: int = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_loss: torch.Tensor = torch.tensor(float('inf'))
        self.detect_posterior_collapse: bool = detect_posterior_collapse

        self.scenario = 'unknown'

        if initial_checkpoint_path is not None and Path(initial_checkpoint_path).exists():
            self._load_args_from_dict(torch.load(initial_checkpoint_path)['args'])
            self.run_name = self._get_run_name(base_run_name)
            self.checkpoint_path = runs_dir / self.run_name / 'checkpoint.pth'
            if self.checkpoint_path.exists():
                if resume_if_exists:
                    self.scenario = 'resume_from_checkpoint'
                else:
                    self.scenario = 'resume_from_other'
            else:
                self.scenario = 'resume_from_other'
        else:
            self._load_args_from_dict(args)
            self.run_name = self._get_run_name(base_run_name)
            self.checkpoint_path = runs_dir / self.run_name / 'checkpoint.pth'
            if resume_if_exists and self.checkpoint_path.exists():
                self.scenario = 'resume_from_checkpoint'
            else:
                self.scenario = 'start_from_scratch'

        logger.info(f'Scenario: {self.scenario}')

        if self.scenario == 'resume_from_checkpoint':
            logger.info(f'Resuming from existing run {self.run_name}, checkpoint: {self.checkpoint_path}')
            reset_optimizer_val = reset_optimizer is True
            self._resume_from_checkpoint(override_saved_args, model_state_only=False, reset_optimizer=reset_optimizer_val, **args)
        elif self.scenario == 'resume_from_other':
            assert initial_checkpoint_path is not None and Path(initial_checkpoint_path).exists()

            self._load_args_from_dict(args)
            self.run_name = self._get_run_name(base_run_name)
            self.checkpoint_path = runs_dir / self.run_name / 'checkpoint.pth'

            logger.info(f'Copying checkpoint from {initial_checkpoint_path} to {self.checkpoint_path}')
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_bytes(Path(initial_checkpoint_path).read_bytes())

            logger.info(f'Resuming from other checkpoint, checkpoint: {initial_checkpoint_path}')
            reset_optimizer_val = reset_optimizer is True or reset_optimizer == 'only_at_import'
            self._resume_from_checkpoint(override_saved_args, model_state_only=True, reset_optimizer=reset_optimizer_val, **args)
        elif self.scenario == 'start_from_scratch':
            logger.info(f'Creating new run {self.run_name}')
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._start_from_scratch()
        else:
            logger.error(f'Unknown scenario: {self.scenario}')
            raise ValueError(f'Unknown scenario: {self.scenario}')

        self._setup_mnist(data_root)
        self.writer = set_up_tensorboard(runs_dir, self.run_name)

        # print summary to log
        logger.info(f'Run name: {self.run_name}')
        logger.info(f'Arguments: {self._get_args_dict()}')
        logger.info(f'Checkpoint path: {self.checkpoint_path}')
        logger.info(f'Device: {self.device}')
        logger.info(f'Starting epoch: {self.start_epoch}')
        logger.info(f'Best loss so far: {self.best_loss}')

    def _setup_mnist(self, data_root) -> None:
        """Set up the MNIST dataset and DataLoaders."""
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.dataset = datasets.MNIST(root=data_root, train=True,
                                      transform=self.transform, download=True)
        # implement 80/20 splitting
        n = len(self.dataset)
        n_train = int(0.8 * n)
        n_test = n - n_train
        dataset_train, dataset_test = torch.utils.data.random_split(self.dataset, [n_train, n_test])

        self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

    def _load_args_from_dict(self, args) -> None:
        """Load arguments from a dictionary."""
        self.original_run_name = cast(Optional[str], args['run_name'] if 'run_name' in args else DEFAULT_ARGS['run_name'])
        self.batch_size = args['batch_size'] if 'batch_size' in args else DEFAULT_ARGS['batch_size']
        self.embed_factor = args['embed_factor'] if 'embed_factor' in args else DEFAULT_ARGS['embed_factor']
        self.beta = args['beta'] if 'beta' in args else DEFAULT_ARGS['beta']
        self.beta_max = args['beta_max'] if 'beta_max' in args else DEFAULT_ARGS['beta_max']
        self.beta_annealing_period = args['beta_annealing_period'] if 'beta_annealing_period' in args else DEFAULT_ARGS['beta_annealing_period']
        self.likelihood_type = args['likelihood_type'] if 'likelihood_type' in args else DEFAULT_ARGS['likelihood_type']
        self.hidden_dim = args['hidden_dim'] if 'hidden_dim' in args else DEFAULT_ARGS['hidden_dim']
        self.expand_dim_enc = args['expand_dim_enc'] if 'expand_dim_enc' in args else DEFAULT_ARGS['expand_dim_enc']
        self.expand_dim_dec = args['expand_dim_dec'] if 'expand_dim_dec' in args else DEFAULT_ARGS['expand_dim_dec']
        self.learning_rate = args['learning_rate'] if 'learning_rate' in args else DEFAULT_ARGS['learning_rate']
        self.num_epochs = args['num_epochs'] if 'num_epochs' in args else DEFAULT_ARGS['num_epochs']
        self.type_of_embedding = args['type_of_embedding'] if 'type_of_embedding' in args else DEFAULT_ARGS['type_of_embedding']
        self.model_version = args['model_version'] if 'model_version' in args else DEFAULT_ARGS['model_version']

    def _get_args_dict(self) -> Dict[str, Any]:
        """Return the current arguments as a dictionary."""
        return {
            'run_name': self.run_name,
            'batch_size': self.batch_size,
            'embed_factor': self.embed_factor,
            'beta': self.beta,
            'beta_max': self.beta_max,
            'beta_annealing_period': self.beta_annealing_period,
            'likelihood_type': self.likelihood_type,
            'hidden_dim': self.hidden_dim,
            'expand_dim_enc': self.expand_dim_enc,
            'expand_dim_dec': self.expand_dim_dec,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'type_of_embedding': self.type_of_embedding,
            'model_version': self.model_version
        }

    def _create_model(self) -> None:
        """Create the VAE model, optimizer, and loss function."""
        input_dims = 1
        if self.type_of_embedding == 'cat':
            input_dims = 2
        elif self.type_of_embedding == 'one_hot':
            input_dims = 11

        self.model = ImageVae(device=self.device,
                              hidden_dim=self.hidden_dim,
                              expand_dim_enc=self.expand_dim_enc,
                              expand_dim_dec=self.expand_dim_dec,
                              input_dims=input_dims,
                              output_dims=1,
                              input_size=64,
                              output_size=64,
                              version=self.model_version)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=15, verbose=True, threshold=1e-4)
        self.criterion = ImageVaeLoss(beta=self.beta, likelihood_type=self.likelihood_type)
        self.model = self.model.to(self.device)

        logger.info('Model information: ')
        logger.info(summary(self.model, (input_dims, 64, 64), verbose=0, device=self.device))

    def _get_run_name(self, base_name: str) -> str:
        """Generate a run name based on the base name and current arguments."""
        ver = self.model_version if self.model_version is not None else 'v1'
        return f"{base_name}_v={ver}_lf={self.likelihood_type}_hid={self.hidden_dim}_exp_enc={self.expand_dim_enc}_exp_dec={self.expand_dim_dec}" \
            f"_embed={self.type_of_embedding}_bmax={self.beta_max}"

    def _resume_from_checkpoint(self, override_saved_args: bool, model_state_only: bool, reset_optimizer: bool, **args) -> None:
        """Resume training from a checkpoint."""
        checkpoint = torch.load(self.checkpoint_path)
        read_args = checkpoint['args'].copy()
        if override_saved_args:
            read_args.update(args)
        self._load_args_from_dict(read_args)
        if model_state_only:
            self.start_epoch = 0
            self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.start_epoch = checkpoint['epoch'] + 1
            self._create_model()
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            try:
                if not reset_optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    logger.debug('Resetting optimizer')
            except BaseException as e:
                logger.warning(f'Could not load optimizer state dict from checkpoint: {e}')

            try:
                self.criterion.load_state_dict(checkpoint['criterion_state_dict'] if 'criterion_state_dict' in checkpoint else checkpoint['loss'])
                if override_saved_args:
                    self.criterion.beta.fill_(self.beta)
            except BaseException as e:
                logger.warning(f'Could not load criterion state dict from checkpoint: {e}')

            best_checkpoint = Path(self.checkpoint_path.parent / 'checkpoint_best.pth')
            if best_checkpoint.exists():
                bcp = torch.load(best_checkpoint)
                self.best_loss = bcp['test_loss'] if 'test_loss' in bcp else float('inf')
            if self.best_loss == float('inf'):
                self.best_loss = checkpoint['test_loss'] if 'test_loss' in checkpoint else float('inf')

    def _start_from_scratch(self) -> None:
        """Start training from scratch."""
        self.start_epoch = 0
        self._create_model()

    def _save_checkpoint(self, epoch: int, test_loss: torch.Tensor, is_best: bool, posterior_collapse_mode: bool) -> None:
        """Save a checkpoint of the current model state."""
        path = self.checkpoint_path
        if posterior_collapse_mode:
            path = path.parent / 'checkpoint_posterior_collapse.pth'
        if is_best:
            path = path.parent / 'checkpoint_best.pth'

        logger.info(f'Saving checkpoint at {path}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'test_loss': test_loss.item(),
            'args': self._get_args_dict(),
            'posterior_collapse': posterior_collapse_mode
        }, path)

        if is_best:
            self.checkpoint_path.write_bytes(path.read_bytes())
            logger.info(f'New best model saved at {path}')

    def _generate_test_samples(self, epoch) -> None:
        """Generate test samples and log them to TensorBoard."""
        self.model.eval()
        with torch.no_grad():
            sample = (self.model.generate(16)*255.0).to(torch.uint8).detach()
            grid = vutils.make_grid(sample, nrow=4, normalize=False).unsqueeze(0)
            self.writer.add_images('Generated samples', grid, epoch)

    def train(self) -> None:
        """The main training loop for the VAE model."""
        # Training loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # summarize current parameters
            logger.info(f'Epoch {epoch+1}/{self.num_epochs}, Beta: {self.criterion.beta.item()}, Learning rate: {self.optimizer.param_groups[0]["lr"]}')

            self.model.train()
            losses = []
            mses = []
            klds = []
            for batch in tqdm(self.dataloader_train, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                x, targets = batch
                x = x.to(self.device)
                targets = targets.to(self.device)
                embed = embed_labels_into_input(x, targets, self.embed_factor, self.type_of_embedding)
                self.optimizer.zero_grad()
                x_hat, z_mean, z_logv = self.model(embed)
                loss, mse, kld = self.criterion(x, x_hat, z_mean, z_logv, self.model.sigma_x)
                loss.backward()

                losses.append(loss)
                mses.append(mse)
                klds.append(kld)

                self.optimizer.step()

            # update beta annealing
            self.criterion.beta.fill_(torch.min(torch.tensor(self.beta_max, device=self.device),
                                      self.criterion.beta + self.beta_max/self.beta_annealing_period))

            avg_loss = torch.stack(losses).mean().item()
            avg_mse = torch.stack(mses).mean().item()
            avg_kld = torch.stack(klds).mean().item()
            logger.info(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss}, Rec loss: {avg_mse}, KL div: {avg_kld}')

            # Log loss to tensorboard
            self.writer.add_scalar('Training loss', avg_loss, epoch)
            self.writer.add_scalar('Rec loss', avg_mse, epoch)
            self.writer.add_scalar('KL div', avg_kld, epoch)
            self.writer.add_scalar('Beta', self.criterion.beta, epoch)
            self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Test loss
            test_loss, x_hat = test_epoch(self.model, self.dataloader_test, self.criterion, self.embed_factor, self.type_of_embedding)
            logger.info(f'Test Loss: {test_loss.item()}')
            self.writer.add_scalar('Test loss', test_loss.item(), epoch)

            self.scheduler.step(test_loss)

            # Check if current model is best (unless beta annealing is active)
            is_best = False
            if self.criterion.beta >= self.beta_max:
                is_best = bool(test_loss < self.best_loss)
                if is_best:
                    self.best_loss = test_loss

            # Detect posterior collapse
            posterior_collapse = False
            if self.detect_posterior_collapse:
                if avg_kld < 1e-3:
                    posterior_collapse = True
                    logger.warning('Posterior collapse detected')

            # Save checkpoint
            self._save_checkpoint(epoch, test_loss, is_best, posterior_collapse)

            # write reconstructed sample to tensorboard
            x_hat_rec = (x_hat*255.0).to(torch.uint8)
            self.writer.add_images('Reconstructed samples', x_hat_rec, epoch)

            # generate 4 samples and write them as images to tensorboard
            self._generate_test_samples(epoch)


def reset_optimizer_param_conv(value: str) -> Union[bool, str]:
    if value.lower() in {'true', 'yes', '1'}:
        return True
    elif value.lower() in {'false', 'no', '0'}:
        return False
    elif value.lower() == 'only_at_import':
        return 'only_at_import'
    else:
        raise argparse.ArgumentTypeError(f'Invalid value for reset_optimizer: {value}')


def main():
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('base_run_name', type=str, help='The base name for the run')
    parser.add_argument('--runs_dir', type=Path, default=Path('runs'), help='Directory to save the runs')
    parser.add_argument('--resume_if_exists', action='store_true', help='Whether to resume training if a checkpoint exists')
    parser.add_argument('--override_saved_args', action='store_true', help='Whether to override saved arguments with new ones')
    parser.add_argument('--reset_optimizer', type=reset_optimizer_param_conv, default=False,
                        help='Whether to reset the optimizer (True, False, or "only_at_import")')
    parser.add_argument('--batch_size', type=int, default=96, help='The batch size for training')
    parser.add_argument('--embed_factor', type=float, default=1.0, help='The embedding factor for the targets')
    parser.add_argument('--beta', type=float, default=0.001, help='The beta parameter for the VAE loss')
    parser.add_argument('--beta_max', type=float, default=1.0, help='The maximum value of beta for annealing')
    parser.add_argument('--beta_annealing_period', type=int, default=20, help='The period over which beta is annealed')
    parser.add_argument('--likelihood_type', type=str, default='bce', help='The type of likelihood used in the VAE loss')
    parser.add_argument('--hidden_dim', type=int, default=32, help='The hidden dimension size for the VAE model')
    parser.add_argument('--expand_dim_enc', type=int, default=32, help='The expansion dimension for the encoder')
    parser.add_argument('--expand_dim_dec', type=int, default=24, help='The expansion dimension for the decoder')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='The learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs for training')
    parser.add_argument('--initial_checkpoint_path', type=str, default=None, help='Path to the initial checkpoint to resume training from')
    parser.add_argument('--data_root', type=Path, default=Path('data'), help='Root directory for the data')
    parser.add_argument('--type_of_embedding', type=str, default='cat', help='Type of embedding for the targets (none, cat, one_hot)')
    parser.add_argument('--model_version', type=str, default=None,
                        choices=[None, 'v1', 'v2'], help='Version of the model (default is None which is identical to v1)')
    parser.add_argument('--detect_posterior_collapse', action='store_true', help='Whether to detect posterior collapse')

    args = vars(parser.parse_args())

    training = VaeTraining(**args)
    training.train()


if __name__ == '__main__':
    main()
