import numpy as np
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm import tqdm, trange
from PIL import Image
from .modules.stransformer import STransformer


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.norm = lambda x: F.normalize(x, dim=-1)
        self.embedding.weight.data.normal_()

        self.re_embed = n_e

    def forward(self, z):
        z_flattened_norm = self.norm(z.reshape(-1, self.e_dim))
        embedding_norm = self.norm(self.embedding.weight)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm**2, dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened_norm, embedding_norm)

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach()-z_norm)**2) + \
                torch.mean((z_qnorm - z_norm.detach()) ** 2)

        # preserve gradients
        z_qnorm = z_norm + (z_qnorm - z_norm).detach()

        return z_qnorm, loss, min_encoding_indices

class ImageTokenizer(nn.Module):
    def __init__(self, encoder_config, decoder_config, cb_config, img_size, patch_size):
        super().__init__()
        encoder_config.n_tokens_per_frame = (img_size // patch_size) ** 2
        encoder_config.block_size = encoder_config.n_tokens_per_frame
        encoder_config.vocab_size = None
        decoder_config.n_tokens_per_frame = (img_size // patch_size) ** 2
        decoder_config.block_size = decoder_config.n_tokens_per_frame
        decoder_config.vocab_size = None
        self.encoder = STransformer(encoder_config)
        self.decoder = STransformer(decoder_config)
        self.to_patch_embed = nn.Sequential(
            nn.Conv2d(3, encoder_config.n_embed, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.encoder_output_proj = nn.Linear(encoder_config.n_embed, cb_config.embed_dim, bias=encoder_config.bias)
        self.decoder_input_proj = nn.Linear(cb_config.embed_dim, decoder_config.n_embed, bias=decoder_config.bias)
        self.to_pixel = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=img_size//patch_size, w=img_size//patch_size),
            nn.ConvTranspose2d(decoder_config.n_embed, 3, kernel_size=patch_size, stride=patch_size),
        )
        self.quantizer = VectorQuantizer(cb_config.n_embed, cb_config.embed_dim, cb_config.beta)
        self.img_size = img_size
        self.patch_size = patch_size

    def encode(self, x):
        # x shape (B C H W)
        x = self.to_patch_embed(x).contiguous()
        h = self.encoder_output_proj(self.encoder(x))
        quant, embed_loss, info = self.quantizer(h)
        return quant, embed_loss, info
    
    def decode(self, quant):
        B = quant.shape[0]
        dec = self.decoder(self.decoder_input_proj(quant)) # (B, H*W, C)
        dec = self.to_pixel(dec).contiguous() # (B, C, H, W)
        dec = dec.reshape(B, -1, *dec.shape[-2:]) # (B, C, H, W)
        return dec
    
    def forward(self, x):
        quant, embed_loss, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, embed_loss
    
    def criterion(self, x, output, loss_form: str='MAE'):
        xrec, qloss = output

        if loss_form == "MAE":
            rec_loss = torch.mean(torch.abs((x - xrec)))
        elif loss_form == "MSE":
            rec_loss = torch.mean((x - xrec) ** 2)
        else:
            raise NotImplementedError
        codebook_loss = qloss.mean()
        loss = rec_loss + codebook_loss

        return loss, {
            'train/loss': loss.item(),
            'train/rec_loss': rec_loss.item(),
            'train/codebook_loss': codebook_loss.item()
        }
        
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

    # def training_step(self, batch):
    #     x, _ = batch
    #     xrec, qloss = self(x)

    #     rec_loss = torch.abs(x - xrec).mean()
    #     codebook_loss = qloss.mean()
    #     loss = rec_loss + codebook_loss

    #     return loss, {
    #         'train/loss': loss.item(),
    #         'train/rec_loss': rec_loss.item(),
    #         'train/codebook_loss': codebook_loss.item()
    #     }
    
    # @torch.no_grad()
    # def validation_step(self, batch):
    #     x, _ = batch
    #     xrec, qloss = self(x)

    #     rec_loss = torch.abs(x - xrec).mean()
    #     codebook_loss = qloss.mean()
    #     loss = rec_loss + codebook_loss

    #     return {
    #         'val/loss': loss.item(),
    #         'val/rec_loss': rec_loss.item(),
    #         'val/codebook_loss': codebook_loss.item()
    #     }
    
    @torch.no_grad()
    def visualize(self, x, save_dir=None):
        origin_imgs = x.clone()
        B, C, H, W = x.shape
        quant, _, encoding_indices = self.encode(x)
        xrec = self.decode(quant)

        rec_imgs = rearrange(xrec, 'b c h w -> c h (b w)').contiguous()
        origin_imgs = rearrange(origin_imgs, 'b c h w -> c h (b w)').contiguous()

        cat_imgs = torch.cat([origin_imgs, rec_imgs], dim=-2).cpu().numpy() # (B, C, 2*H, T*W)
        cat_imgs = ((cat_imgs + 1) * 127.5).clip(0, 255).astype(np.uint8)
        cat_imgs = cat_imgs.transpose(1, 2, 0)

        if save_dir is not None:
            for i in range(cat_imgs.shape[0]):
                Image.fromarray(cat_imgs[i]).save(save_dir+f'/vis_{i}.png')
        else:
            return cat_imgs, encoding_indices.cpu().numpy()
        
