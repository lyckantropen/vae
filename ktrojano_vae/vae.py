from logging import getLogger
from pathlib import Path
from typing import Tuple

import torch

logger = getLogger(Path(__file__).stem)


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

    Attributes
    ----------
    input_size : int
        The size of the input image.
    latent_size : int
        The size of the latent representation.
    expand_dim : int
        The number of filters in the convolutional layers.
    hidden_dim : int
        The dimension of the hidden layer.
    cnn1 : torch.nn.Conv2d
        The first convolutional layer.
    cnn2 : torch.nn.Conv2d
        The second convolutional layer.
    cnn3 : torch.nn.Conv2d
        The third convolutional layer.
    dense : torch.nn.Linear
        The fully connected layer.

    Methods
    -------
    _initialize_weights()
        Initializes the weights of the network.
    forward(x)
        Defines the forward pass of the network.
    """

    def __init__(self, hidden_dim, input_dims=3, input_size=64, expand_dim=32) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_size = self.input_size >> 3
        self.expand_dim = expand_dim
        self.hidden_dim = hidden_dim

        self.cnn1 = torch.nn.Conv2d(input_dims, expand_dim, kernel_size=3, stride=2, padding=1)
        self.cnn2 = torch.nn.Conv2d(expand_dim, expand_dim, kernel_size=3, stride=2, padding=1)
        self.cnn3 = torch.nn.Conv2d(expand_dim, expand_dim, kernel_size=3, stride=2, padding=1)
        self.dense = torch.nn.Linear(self.latent_size*self.latent_size*expand_dim, hidden_dim*2)

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.kaiming_normal_(self.cnn1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn3.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.dense.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.cnn1(x))
        x = torch.relu(self.cnn2(x))
        x = torch.relu(self.cnn3(x))
        x = x.view(x.size(0), -1)
        x = self.dense(x)

        # split the vector in half
        return torch.relu(x[:, :self.hidden_dim]), torch.relu(x[:, self.hidden_dim:])


class Decoder(torch.nn.Module):
    """
    A Decoder class for a Variational Autoencoder (VAE) that reconstructs images from latent space representations.

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

    Attributes
    ----------
    output_size : int
        The spatial size of the output image.
    latent_size : int
        The spatial size of the latent representation.
    expand_dim : int
        The number of feature maps in the intermediate layers.
    dense : torch.nn.Linear
        Fully connected layer to expand the latent space representation.
    cnn1 : torch.nn.ConvTranspose2d
        First transposed convolutional layer.
    cnn2 : torch.nn.ConvTranspose2d
        Second transposed convolutional layer.
    cnn3 : torch.nn.ConvTranspose2d
        Third transposed convolutional layer.

    Methods
    -------
    _initialize_weights()
        Initializes the weights of the network using Kaiming normal initialization.
    forward(x)
        Defines the forward pass of the decoder.
    """

    def __init__(self, hidden_dim, output_dims=3, output_size=64, expand_dim=32, final_activation='sigmoid') -> None:
        super().__init__()
        self.output_size = output_size
        self.latent_size = self.output_size >> 3
        self.expand_dim = expand_dim

        self.dense = torch.nn.Linear(hidden_dim, expand_dim*self.latent_size*self.latent_size)
        self.cnn1 = torch.nn.ConvTranspose2d(expand_dim, expand_dim, kernel_size=4, stride=2, padding=1)
        self.cnn2 = torch.nn.ConvTranspose2d(expand_dim, expand_dim, kernel_size=4, stride=2, padding=1)
        self.cnn3 = torch.nn.ConvTranspose2d(expand_dim, output_dims, kernel_size=4, stride=2, padding=1)
        self.final_activation: torch.nn.Module
        if final_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation == 'hardtanh':
            self.final_activation = torch.nn.Hardtanh(0.0, 1.0)
        else:
            raise ValueError(f'Unknown final activation: {final_activation}')

        self._initialize_weights()

    def _initialize_weights(self):
        torch.nn.init.kaiming_normal_(self.dense.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.cnn3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = torch.relu(self.dense(x))
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = torch.nn.functional.pixel_shuffle(x, self.latent_size)
        x = torch.relu(self.cnn1(x))
        x = torch.relu(self.cnn2(x))
        x = self.final_activation(self.cnn3(x))
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

    Attributes
    ----------
    beta : torch.nn.Parameter
        The weight of the KL divergence term.
    likelihood_type : str
        The type of likelihood to use.
    """

    def __init__(self, beta=1.0, likelihood_type='mse') -> None:
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
        if kl_divergence < 1e-9:
            logger.warning('KL divergence is close to zero, possible posterior collapse.')

        return neg_log_likelihood + self.beta.to(x.device)*kl_divergence, neg_log_likelihood, kl_divergence


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

    Methods
    -------
    forward(x)
        Performs a forward pass through the VAE.
    reparameterize(z_mean, z_logv)
        Reparameterizes the latent variables using the mean and log variance.
    generate()
        Generates a new image by sampling from the latent space.
    """

    def __init__(self, device, hidden_dim, expand_dim_enc=32, expand_dim_dec=16, input_size=64, output_size=64, input_dims=3, output_dims=3, sigma_x=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.encoder = Encoder(hidden_dim, expand_dim=expand_dim_enc, input_dims=input_dims, input_size=input_size)
        self.decoder = Decoder(hidden_dim, expand_dim=expand_dim_dec, output_dims=output_dims, output_size=output_size)
        self.sigma_x = torch.tensor(sigma_x, device=device, requires_grad=False)

    def forward(self, x):
        z_mean, z_logv = self.encoder(x)
        z = self.reparameterize(z_mean, z_logv)
        return self.decoder(z), z_mean, z_logv

    def reparameterize(self, z_mean, z_logv):
        z_std = torch.exp(0.5 * z_logv)
        z = z_mean + z_std * torch.randn_like(z_mean, device=self.device)
        return z

    def generate(self, num_samples=1):
        with torch.no_grad():
            z = torch.randn((num_samples, self.hidden_dim), device=self.device)
            return torch.clip(self.decoder(z), 0, 1)
