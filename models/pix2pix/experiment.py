import os
import time

import numpy as np
import torch
from torch import  nn, optim
from torchvision.utils import save_image


def trainer(epochs, generator, discriminator, train_loader, val_loader=None, save_fig_dir=None, device='cpu'):
    gen = generator.to(device)
    disc = discriminator.to(device)

    batch_size = train_loader.batch_size
    n_batch = len(train_loader)

    opt_gen = optim.Adam(gen.parameters(), lr=3e-4)
    opt_disc = optim.Adam(disc.parameters(), lr=3e-4)
    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        start_time = time.time()
        disc.train()
        gen.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            y_fake = gen(x)

            # train disc: max log(D(real)) + log(1 - D(G(z)))
            disc_real = disc(x,y)
            disc_fake = disc(x,y_fake)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            opt_disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # train gen: min log(1 - D(G(z))) <-> max log(D(G(z)))
            output = disc(x, y_fake)
            lossG = criterion(output, torch.ones_like(output)) + 100*l1_loss(y_fake, y)
            opt_gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            print(f'\r{batch_idx+1}/{n_batch} - lossD: {lossD:.4f} - lossG: {lossG:.4f}',end='')
        
        end_time = time.time()
        delta_time = end_time - start_time
        print(f' - {delta_time:.0f}s/epoch')

        if val_loader != None and save_fig_dir != None:
            x,y = next(iter(val_loader))
            x,y = x.to(device), y.to(device)
            gen.eval()
            y_fake = gen(x)

            save_image(torch.cat([x,y_fake,y],dim=-1),
                os.path.join(save_fig_dir,f'{epoch}.png'))




def main():
    from dataset import AnimeDataset
    from discriminator import Discriminator
    from generator import Generator
    from torch.utils.data import DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = AnimeDataset('data/AnimeData/train/')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)

    val_dataset = AnimeDataset('data/AnimeData/val/')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    gen = Generator().to(device)
    dist = Discriminator().to(device)

    trainer(2, gen, dist, train_loader, val_loader, save_fig_dir='figs', device=device)


if __name__ == '__main__':
    main()
