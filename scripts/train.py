import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ktrojano_vae.vae import ImageVae, ImageVaeLoss

# defaults
RUN_NAME = 'mnist_bh'
BATCH_SIZE = 92
EMBED_FACTOR = 0.0
BETA = 0.001
BETA_MAX = 0.4
BETA_ANNEALING_PERIOD = 20


class PadToMultipleOf16(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        _, h, w = img.shape
        new_w = (w + 15) // 16 * 16
        new_h = (h + 15) // 16 * 16
        # pad symmetrically
        pad_w = (new_w - w) // 2
        pad_h = (new_h - h) // 2
        img = transforms.functional.pad(img, (pad_w, pad_h, pad_w, pad_h))
        return img


def set_up_tensorboard(run_name):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(f'runs/{run_name}')
    except ImportError:
        print('Tensorboard not available')
        return None


def test_epoch(model, dataloader, criterion, embed_factor):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            x, targets = batch
            x = x.to(model.device)
            targets = targets.to(model.device)
            # embed the targets into the input
            embed = torch.cat([x, embed_factor*targets.view(-1, 1, 1, 1).repeat(1, 1, x.size(-2), x.size(-1))], dim=1)
            x_hat, z_mean, z_logv = model(embed)
            loss, _, _ = criterion(x, x_hat, z_mean, z_logv, model.sigma_x)
            losses.append(loss)
    return torch.stack(losses).mean(), x_hat[0:1].detach()


def train():
    args = {
        'run_name': 'mnist_bce',
        'batch_size': BATCH_SIZE,
        'embed_factor': EMBED_FACTOR,
        'beta': BETA,
        'beta_max': BETA_MAX,
        'beta_annealing_period': BETA_ANNEALING_PERIOD,
        'likelihood_type': 'bce',
    }
    writer = set_up_tensorboard(args['run_name'])

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST(root='C:/Users/troja/data', train=True,
                             transform=transform, download=True)
    # implement 80/20 splitting
    n = len(dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [n_train, n_test])

    dataloader_train = DataLoader(dataset_train, batch_size=args['batch_size'], shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args['batch_size'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer, and loss function
    model = ImageVae(device=device, hidden_dim=32, expand_dim_enc=32, expand_dim_dec=16, input_dims=2, output_dims=1, input_size=64, output_size=64)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = ImageVaeLoss(beta=args['beta'], likelihood_type=args['likelihood_type'])

    model = model.to(device)

    # Load checkpoint if exists
    checkpoint_path = Path('runs') / args['run_name'] / 'checkpoint.pth'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion.load_state_dict(checkpoint['loss'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    num_epochs = 1000
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = []
        mses = []
        klds = []
        for batch in tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x, targets = batch
            x = x.to(device)
            targets = targets.to(device)
            # embed the targets into the input
            embed = torch.cat([x, args['embed_factor']*targets.view(-1, 1, 1, 1).repeat(1, 1, x.size(-2), x.size(-1))], dim=1)
            optimizer.zero_grad()
            x_hat, z_mean, z_logv = model(embed)
            loss, mse, kld = criterion(x, x_hat, z_mean, z_logv, model.sigma_x)
            loss.backward()

            losses.append(loss)
            mses.append(mse)
            klds.append(kld)

            optimizer.step()

        criterion.beta = torch.min(torch.tensor(args['beta_max'], device=device), criterion.beta + args['beta_max']/args['beta_annealing_period'])

        avg_loss = torch.stack(losses).mean().item()
        avg_mse = torch.stack(mses).mean().item()
        avg_kld = torch.stack(klds).mean().item()
        # Log loss to tensorboard
        if writer is not None:
            writer.add_scalar('Training loss', avg_loss, epoch)
            writer.add_scalar('Rec loss', avg_mse, epoch)
            writer.add_scalar('KL div', avg_kld, epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion.state_dict(),
            'args': args
        }, checkpoint_path)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

        # Test loss
        test_loss, x_hat = test_epoch(model, dataloader_test, criterion, args['embed_factor'])
        print(f'Test Loss: {test_loss.item()}')
        if writer is not None:
            writer.add_scalar('Test loss', test_loss.item(), epoch)
            x_hat_rec = (x_hat*255.0).to(torch.uint8)
            writer.add_images('Reconstructed samples', x_hat_rec, epoch)

        # generate 4 samples and write them as images to tensorboard
        if writer is not None:
            model.eval()
            with torch.no_grad():
                for _ in range(4):
                    sample = (model.generate()*255.0).to(torch.uint8).detach()
                    writer.add_images('Generated samples', sample, epoch)


if __name__ == '__main__':
    train()
