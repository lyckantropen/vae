import torch
from typing import Tuple


class Encoder(torch.nn.Module):
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
    def __init__(self, hidden_dim, output_dims=3, output_size=64, expand_dim=32) -> None:
        super().__init__()
        self.output_size = output_size
        self.latent_size = self.output_size >> 3
        self.expand_dim = expand_dim

        self.dense = torch.nn.Linear(hidden_dim, expand_dim*self.latent_size*self.latent_size)
        self.cnn1 = torch.nn.ConvTranspose2d(expand_dim, expand_dim, kernel_size=4, stride=2, padding=1)
        self.cnn2 = torch.nn.ConvTranspose2d(expand_dim, expand_dim, kernel_size=4, stride=2, padding=1)
        self.cnn3 = torch.nn.ConvTranspose2d(expand_dim, output_dims, kernel_size=4, stride=2, padding=1)

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
        x = torch.nn.functional.hardtanh(self.cnn3(x), 0.0, 1.0)
        # x = torch.sigmoid(self.cnn3(x))
        return x


class ImageVaeLoss(torch.nn.Module):
    def __init__(self, beta=1.0, likelihood_type='mse') -> None:
        super().__init__()
        self.beta = torch.tensor(beta, requires_grad=False)
        self.likelihood_type = likelihood_type

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, z_mean: torch.Tensor, z_logv: torch.Tensor, sigma_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Loss based on NELBO."""
        # scaling to prevent posterior collapse
        if self.likelihood_type == 'mse':
            neg_log_likelihood = 1./(2.*sigma_x*sigma_x)*torch.mean((x - x_hat)**2, dim=(1, 2, 3)).sum() * z_mean.size(1)**2
        elif self.likelihood_type == 'bce':
            neg_log_likelihood = torch.nn.functional.binary_cross_entropy(x_hat, x, reduce=False).mean(dim=(1, 2, 3)).sum() * z_mean.size(1)**2
        else:
            raise ValueError(f'Unknown likelihood type: {self.likelihood_type}')

        # neg_log_likelihood = self.bce_loss(x_hat, x)
        kl_divergence = 1./2. * torch.sum(z_logv.exp() + z_mean.pow(2) - 1 - z_logv)
        # free bits trick
        kl_divergence = torch.maximum(kl_divergence, torch.tensor(float(1/z_mean.size(1)), device=x.device))
        return neg_log_likelihood + self.beta.to(x.device)*kl_divergence, neg_log_likelihood, kl_divergence


class ImageVae(torch.nn.Module):
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

    def generate(self):
        with torch.no_grad():
            z = torch.randn((1, self.hidden_dim), device=self.device)
            return torch.clip(self.decoder(z), 0, 1)
