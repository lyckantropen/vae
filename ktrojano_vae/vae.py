from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = getLogger(Path(__file__).stem)


class SigmoidSymmetric(torch.nn.Module):
    """A sigmoid activation function that maps to the range [-scale, scale]."""

    def __init__(self, scale: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale*2.0 * self.sigmoid(x) - 1.0*self.scale


class Encoder(torch.nn.Module):
    """
    A Convolutional Encoder for Variational Autoencoders (VAEs).

    The first half of the latent space representation is the mean of the
    Gaussian distribution, while the second half is the log-variance.

    Parameters
    ----------
    hidden_dim : int
        The dimension of the hidden layer.
    input_dims : int, optional
        The number of input channels (default is 3).
    input_size : int, optional
        The size of the input image (default is 64).
    expand_dim : int, optional
        The number of filters in the convolutional layers (default is 32).
    final_activation : str, optional
        The final activation function to use, one of 'sigmoid' or 'hardtanh' (default is 'sigmoid').
    version : str, optional
        The version of the encoder to use (default is None).

    Methods
    -------
    _initialize_weights()
        Initializes the weights of the network.
    forward(x)
        Defines the forward pass of the network.
    """

    def __init__(self,
                 hidden_dim: int,
                 input_dims: int = 3,
                 input_size: int = 64,
                 expand_dim: int = 32,
                 final_activation: str = 'sigmoid',
                 version: Optional[str] = None
                 ) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = self.input_size >> 3
        self.expand_dim = expand_dim
        self.hidden_dim = hidden_dim
        self.version = version
        self.final_activation_mu: torch.nn.Module
        self.final_activation_logv: torch.nn.Module

        self.cnn1 = torch.nn.Conv2d(input_dims, expand_dim, kernel_size=3, stride=2, padding=1)
        self.act1 = torch.nn.ReLU()
        self.cnn2 = torch.nn.Conv2d(expand_dim, expand_dim, kernel_size=3, stride=2, padding=1)
        self.act2 = torch.nn.ReLU()
        self.cnn3 = torch.nn.Conv2d(expand_dim, expand_dim, kernel_size=3, stride=2, padding=1)
        self.act3 = torch.nn.ReLU()
        self.dense = torch.nn.Linear(self.latent_size*self.latent_size*expand_dim, hidden_dim*2)
        self.mu_scale = torch.nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.logv_scale = torch.nn.Parameter(torch.ones(1, hidden_dim)*10.0, requires_grad=True)

        if final_activation == 'sigmoid':
            self.final_activation_mu = SigmoidSymmetric()
            self.final_activation_logv = torch.nn.Sigmoid()
        elif final_activation == 'hardtanh':
            self.final_activation_mu = torch.nn.Hardtanh(-1.0, 1.0)
            self.final_activation_logv = torch.nn.Hardtanh(0.0, 1.0)
        else:
            raise ValueError(f'Unknown final activation: {final_activation}')

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        torch.nn.init.kaiming_normal_(self.cnn1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn3.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.dense.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.mu_scale.clamp_(min=1.0, max=100.0)
            self.logv_scale.clamp_(min=1.0, max=100.0)
        x = self.act1(self.cnn1(x))
        x = self.act2(self.cnn2(x))
        x = self.act3(self.cnn3(x))
        x = x.view(x.size(0), -1)
        x = self.dense(x)

        # split the vector in half
        return (self.final_activation_mu(x[:, :self.hidden_dim])*self.mu_scale,
                torch.log(self.final_activation_logv(x[:, self.hidden_dim:])*self.logv_scale + 1e-8))


class Decoder(torch.nn.Module):
    """
    A Decoder class for a Variational Autoencoder (VAE) that reconstructs images from latent space representations.

    Two versions are available: 'v1' uses a single transposed convolutional
    layer to generate the output image, while 'v2' uses two transposed
    convolutional layers followed by a 3x3 convolutional layer. Version 'v2'
    should produce less artifacts.

    Parameters
    ----------
    hidden_dim : int
        The dimensionality of the latent space.
    output_dims : int, optional
        The number of output channels (default is 3, for RGB images).
    output_size : int, optional
        The spatial size of the output image (default is 64).
    expand_dim : int, optional
        The number of feature maps in the intermediate layers (default is 32).
    final_activation : str, optional
        The final activation function to use, one of 'sigmoid' or 'hardtanh' (default is 'sigmoid').
    version : str, optional
        The version of the decoder to use (default is None).

    Methods
    -------
    _initialize_weights()
        Initializes the weights of the network using Kaiming normal initialization.
    forward(x)
        Defines the forward pass of the decoder.
    """

    def __init__(self,
                 hidden_dim: int,
                 output_dims: int = 3,
                 output_size: int = 64,
                 expand_dim: int = 32,
                 final_activation: str = 'sigmoid',
                 version: Optional[str] = None
                 ) -> None:
        super().__init__()
        self.output_size = output_size
        self.latent_size = self.output_size >> 3
        self.expand_dim = expand_dim
        self.version = version

        self.dense = torch.nn.Linear(hidden_dim, expand_dim*self.latent_size*self.latent_size)
        self.act0 = torch.nn.ReLU()
        self.shuffle = torch.nn.PixelShuffle(self.latent_size)
        self.cnn1 = torch.nn.ConvTranspose2d(expand_dim, expand_dim, kernel_size=4, stride=2, padding=1)
        self.act1 = torch.nn.ReLU()
        self.cnn2 = torch.nn.ConvTranspose2d(expand_dim, expand_dim, kernel_size=4, stride=2, padding=1)
        self.act2 = torch.nn.ReLU()

        if version is None or version == 'v1':
            self.cnn3 = torch.nn.ConvTranspose2d(expand_dim, output_dims, kernel_size=4, stride=2, padding=1)
            self.cnn4 = None
        elif version == 'v2':
            self.cnn3 = torch.nn.ConvTranspose2d(expand_dim, output_dims*2, kernel_size=4, stride=2, padding=1)
            self.act3 = torch.nn.ReLU()
            self.cnn4 = torch.nn.Conv2d(output_dims*2, output_dims, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f'Unknown version: {version}')

        self.final_activation: torch.nn.Module
        if final_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation == 'hardtanh':
            self.final_activation = torch.nn.Hardtanh(0.0, 1.0)
        else:
            raise ValueError(f'Unknown final activation: {final_activation}')

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        torch.nn.init.kaiming_normal_(self.dense.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn3.weight, mode='fan_out', nonlinearity='relu')
        if self.cnn4 is not None:
            torch.nn.init.kaiming_normal_(self.cnn4.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.dense(x))
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.shuffle(x)
        x = self.act1(self.cnn1(x))
        x = self.act2(self.cnn2(x))
        x = self.cnn3(x)
        if self.cnn4 is not None:
            x = self.cnn4(self.act3(x))
        x = self.final_activation(x)
        return x


class ImageVaeLoss(torch.nn.Module):
    """
    A custom loss function for Variational Autoencoders (VAEs) that combines
    the negative log-likelihood and the KL divergence, with support for
    different likelihood types.

    Parameters
    ----------
    beta : float, optional
        The weight of the KL divergence term in the loss function. Default is 1.0.
    likelihood_type : str, optional
        The type of likelihood to use. Can be 'mse' for mean squared error or
        'bce' for binary cross-entropy. Default is 'mse'.

    Methods
    -------
    forward(x, x_hat, z_mean, z_logv, sigma_x)
        Computes the VAE loss given the input tensors.
    """

    def __init__(self, beta: float = 1.0, likelihood_type: str = 'mse') -> None:
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(beta, requires_grad=False), requires_grad=False)
        self.likelihood_type = likelihood_type

    def forward(self,
                x: torch.Tensor,
                x_hat: torch.Tensor,
                z_mean: torch.Tensor,
                z_logv: torch.Tensor,
                sigma_x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the VAE loss given the input tensors.

        This is a NELBO-based loss function that combines the negative log-likelihood
        and the KL divergence terms.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        x_hat : torch.Tensor
            The reconstructed tensor.
        z_mean : torch.Tensor
            The mean of the latent space representation.
        z_logv : torch.Tensor
            The log-variance of the latent space representation.
        sigma_x : torch.Tensor
            The standard deviation of the likelihood.
        """
        # scaling by latent size squared to prevent posterior collapse
        if self.likelihood_type == 'mse':
            neg_log_likelihood = 1./(2.*sigma_x*sigma_x)*torch.mean((x - x_hat)**2, dim=(1, 2, 3)).sum() * z_mean.size(1)**2
        elif self.likelihood_type == 'bce':
            neg_log_likelihood = torch.nn.functional.binary_cross_entropy(x_hat, x, reduce=False).mean(dim=(1, 2, 3)).sum() * z_mean.size(1)**2
        else:
            raise ValueError(f'Unknown likelihood type: {self.likelihood_type}')

        kl_divergence = 1./2. * torch.sum(z_logv.exp() + z_mean.pow(2) - 1 - z_logv)
        if kl_divergence < 1e-3:
            logger.warning('KL divergence is close to zero, possible posterior collapse.')

        return neg_log_likelihood + self.beta*kl_divergence, neg_log_likelihood, kl_divergence


class ImageVae(torch.nn.Module):
    """
    Variational Autoencoder (VAE) for image data.

    Parameters
    ----------
    device : torch.device
        The device (CPU or GPU) on which the model will be run.
    hidden_dim : int
        The dimensionality of the latent space.
    expand_dim_enc : int, optional
        The expansion dimension for the encoder, by default 32.
    expand_dim_dec : int, optional
        The expansion dimension for the decoder, by default 16.
    input_size : int, optional
        The size of the input images, by default 64.
    output_size : int, optional
        The size of the output images, by default 64.
    input_dims : int, optional
        The number of input channels, by default 3.
    output_dims : int, optional
        The number of output channels, by default 3.
    sigma_x : float, optional
        The standard deviation of the Gaussian distribution for the output, by default 1.0.
    version : str, optional
        The version of the model to use, by default None.

    Methods
    -------
    forward(x)
        Performs a forward pass through the VAE.
    reparameterize(z_mean, z_logv)
        Reparameterizes the latent variables using the mean and log variance.
    generate()
        Generates a new image by sampling from the latent space.
    """

    def __init__(self,
                 device: torch.device,
                 hidden_dim: int,
                 expand_dim_enc: int = 32,
                 expand_dim_dec: int = 16,
                 input_size: int = 64,
                 output_size: int = 64,
                 input_dims: int = 3,
                 output_dims: int = 3,
                 sigma_x: float = 1.0,
                 version: Optional[str] = None
                 ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = Encoder(hidden_dim, expand_dim=expand_dim_enc, input_dims=input_dims, input_size=input_size, version=version)
        self.decoder = Decoder(hidden_dim, expand_dim=expand_dim_dec, output_dims=output_dims, output_size=output_size, version=version)
        self.sigma_x = torch.tensor(sigma_x, device=device, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_logv = self.encoder(x)
        z = self.reparameterize(z_mean, z_logv)
        return self.decoder(z), z_mean, z_logv

    def reparameterize(self, z_mean: torch.Tensor, z_logv: torch.Tensor) -> torch.Tensor:
        z_std = torch.exp(0.5 * z_logv)
        z = z_mean + z_std * torch.randn_like(z_mean, device=self.device)
        return z

    def generate(self, num_samples: int = 1) -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn((num_samples, self.hidden_dim), device=self.device)
            return torch.clip(self.decoder(z), 0, 1)
