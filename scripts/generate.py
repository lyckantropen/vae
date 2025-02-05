import argparse
import logging
from pathlib import Path

import torch
from torchvision.utils import save_image

from ktrojano_vae.vae import ImageVae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__file__).stem)


def generate_samples(checkpoint_path: str, num_samples: int, output_path: str, grid_size: int) -> None:
    """Generate samples using the trained model and save them as a grid."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    # Warn about posterior collapse
    if checkpoint['posterior_collapse']:
        logger.warning('Posterior collapse detected. Generated samples may not be diverse.')

    # Create model
    model = ImageVae(device=device,
                     hidden_dim=args['hidden_dim'],
                     expand_dim_enc=args['expand_dim_enc'],
                     expand_dim_dec=args['expand_dim_dec'],
                     input_dims=2,
                     output_dims=1,
                     input_size=64,
                     output_size=64)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Generate samples
    with torch.no_grad():
        samples = model.generate(num_samples).cpu()

    # Save samples as a grid
    save_image(samples, output_path, nrow=grid_size, normalize=True)
    logger.info(f'Samples saved to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='Path to the model checkpoint', required=True)
    parser.add_argument('--num_samples', type=int, help='Number of samples to generate', default=64)
    parser.add_argument('--output_path', type=str, help='Path to save the generated samples', default='samples.png')
    parser.add_argument('--grid_size', type=int, default=8, help='Grid size for the generated samples')

    args = parser.parse_args()

    generate_samples(args.checkpoint_path, args.num_samples, args.output_path, args.grid_size)


if __name__ == '__main__':
    main()
