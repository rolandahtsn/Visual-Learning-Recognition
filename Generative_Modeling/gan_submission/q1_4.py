import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    b_real_ones = torch.full_like(discrim_real,1.0,dtype=torch.float)
    a_fake_zeros = torch.full_like(discrim_fake,0.0,dtype=torch.float)
    # print(discrim_real)
    # print(discrim_fake)
    # Dx_sig_real = torch.sigmoid(discrim_real) 
    # DGz_sig_fake = torch.sigmoid(discrim_fake) 
    Dx_sig_real = (discrim_real) 
    DGz_sig_fake = (discrim_fake) 
    # print(Dx_sig_real)
    # print(DGz_sig_fake)
    loss1 = (Dx_sig_real - b_real_ones)**2
    loss2 = (DGz_sig_fake - a_fake_zeros)**2

    loss = 0.5*(torch.mean(loss1) + torch.mean(loss2))
    # loss = loss.mean()
    # loss = loss/2
    print("discriminator loss", loss)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    DGz_sig_fake = (discrim_fake)
    loss = (DGz_sig_fake - 1.0)**2
    loss = torch.mean(loss)
    loss = loss/2
    print("generator loss", loss)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
