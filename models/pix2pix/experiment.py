import os
import time

import numpy as np
import torch
from torch import  nn, optim
from torchvision.utils import save_image


def trainer(epochs, generator, discriminator, train_loader, val_loader=None, save_fig_dir=None, device='cpu'):
    gen = generator.to(device)
    disc = discriminator.to(device)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    batch_size = train_loader.batch_size
    n_batch = len(train_loader)

    opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        start_time = time.time()
        disc.train()
        gen.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            with torch.cuda.amp.autocast():
                # generate fake colored imgs
                y_fake = gen(x)
                # validate real and fake imgs using discriminator
                D_real = disc(x, y)
                D_fake = disc(x, y_fake.detach())
                # calculate fake and real discriminator loss
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                # calculate average discriminator loss of fake and real colored image
                D_loss = (D_real_loss + D_fake_loss) / 2
            # update weight
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            with torch.cuda.amp.autocast():
                D_fake = disc(x, y_fake)
                G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
                L1 = l1(y_fake, y) * 100
                G_loss = G_fake_loss + L1
            opt_gen.zero_grad()
            d_scaler.scale(G_loss).backward()
            d_scaler.step(opt_gen)
            d_scaler.update()

            print(f'\r{batch_idx+1}/{n_batch} - lossD: {D_loss:.4f} - lossG: {G_loss:.4f}',end='')
        
        end_time = time.time()
        delta_time = end_time - start_time
        print(f' - {delta_time:.0f}s/epoch')

        if val_loader != None and save_fig_dir != None:
            x,y = next(iter(val_loader))
            x,y = x.to(device), y.to(device)
            gen.eval()
            y_fake = gen(x)

            save_image(torch.cat([y_fake,y],dim=-1)*0.5+0.5,
                os.path.join(save_fig_dir,f'{epoch}.png'))