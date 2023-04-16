import numpy as np
import time

import torch
from torch import  nn, optim


def trainer(epochs, generator, discriminator, loader, critic_iteration=5, device='cpu'):
    gen = generator.to(device)
    disc = discriminator.to(device)

    batch_size = loader.batch_size
    n_batch = len(loader)
    z_dim = gen.z_dim

    opt_gen = optim.Adam(gen.parameters(), lr=3e-4)
    opt_disc = optim.Adam(disc.parameters(), lr=3e-4)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        start_time = time.time()
        disc.train()
        gen.train()

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)  # (32,1,28,28)
            
            for _ in range(critic_iteration):
                noise = torch.randn((real.size(0), z_dim)).to(device) # (32,100)
                fake = gen(noise) # (32,1,28,28)

                # train disc: max log(D(real)) + log(1 - D(G(z)))
                disc_real = disc(real).view(-1) # 32
                disc_fake = disc(fake).view(-1) # 32
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2
                opt_disc.zero_grad()
                lossD.backward(retain_graph=True)
                opt_disc.step()

            # train gen: min log(1 - D(G(z))) <-> max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            opt_gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            print(f'\r{batch_idx+1}/{n_batch} - lossD: {lossD:.4f} - lossG: {lossG:.4f}',end='')
        
        end_time = time.time()
        delta_time = end_time - start_time
        print(f' - {delta_time:.0f}s/epoch')