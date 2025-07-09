import torch
import torch.nn as nn
from torch import Tensor
from models.backbones.mae import MaskedAutoencoderViT
from models.criterion import PartialLikelihood
from typing import Type
from tensordict import TensorDict
from rich import print
from functools import partial

from torchvision.models.resnet import resnet50


class CNNSTRAP(nn.Module):
    """ A simple masked convolutional neural network for survival analysis.
    
    Args:
        embed_dim (int): Dimension of the embedding. Default: 1024.
        n_tissues (int): Number of tissues to predict. Default: 3.
        n_factors (int): Number of risk factors. Default: 0.
        pretrained_resnet (bool): Whether to use pretrained ResNet model. Default: True.
    """

    def __init__(self, 
                 embed_dim: int = 1024,
                 n_tissues: int = 3, 
                 n_factors: int = 0,
                 pretrained_resnet: bool = True
                 ):
        
        super().__init__()
        self.n_tissues = n_tissues
        self.embed_dim = embed_dim
        self.n_factors = n_factors
        self.pretrained_resnet = pretrained_resnet

        self.resnet = resnet50(pretrained=pretrained_resnet)

        # Create a linear layer to adapt the features 
        # based on the last layer of the resnet50 model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(n_factors, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        self.adapter    = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, 1),
        )

        # Create the loss function
        self.cox_loss = PartialLikelihood()


    def forward(self,
                images: Tensor,
                masks: Tensor,
                factors: Tensor,
                time: Tensor | None = None,
                event: Tensor | None = None,
                ) -> TensorDict:
        """
        Forward pass of the ConvDeepSurv model.

        Args:
            images (Tensor): Input tensor of shape (B, C, H, W).
            masks (Tensor): Mask tensor of shape (B, H, W).
            time (Tensor | None): Survival times. Default: None.
            event (Tensor | None): Event indicators. Default: None.
            factors (Tensor | None): Additional factors. Default: None.
        Returns:
            TensorDict: A dictionary containing the Cox loss, total loss, hazards,
                        predictions, factors, time, and event.
        """

        # Multiply the masks by the images
        images = images * masks

        images = images.view(-1, 1, images.shape[2], images.shape[3]).repeat(1, 3, 1, 1)

        # Pass the images through the resnet model
        latent = self.resnet(images)

        # Repeat the risk factors for each tissue
        if self.n_factors > 0 and factors is not None:
            factors = factors.repeat_interleave(self.n_tissues, dim=0).squeeze()

            factors = self.fc(factors) # (B, embed_dim)
        
        fs = torch.cat([latent, factors], dim=-1) if factors is not None else latent
        
        hazards: Tensor = self.adapter(fs)

        # Sum the hazards over the tissues
        hazards = hazards.view(-1, self.n_tissues).mean(dim=1)

        # Check if longer than 1, else squeeze
        if hazards.shape[0] > 1:
            hazards = hazards.squeeze()

        # Compute the Cox partial log likelihood
        if time is not None and event is not None:
            cox_loss = self.cox_loss(hazards, time.squeeze(), event.squeeze())
        else:
            cox_loss = torch.tensor(0.0, device=factors.device)
            time = torch.tensor(-1.0, device=factors.device)
            event = torch.tensor(-1.0, device=factors.device)


        output = TensorDict({
            "cox_loss": cox_loss,
            "loss": cox_loss,
            "hazards": hazards.detach(),
            "factors": factors.detach(),
            "time": time.detach(),
            "event": event.detach(),
        })

        return output


class MAE(nn.Module):

    def __init__(self, 
                 img_size: int = 224, 
                 patch_size: int = 16, 
                 in_chans: int = 1, 
                 embed_dim: int = 768, 
                 depth: int = 12, 
                 num_heads: int = 12, 
                 mlp_ratio: float = 4.0,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 mask_ratio: float = 0.75,
                 pretrained: bool = False):
        """ Masked Autoencoder for STRAP. Does not depend on survival analysis,
        and is only used for pretraining the model on images.
        
        Args:
            img_size (int): Size of the input image. Default: 224.
            patch_size (int): Size of the patches. Default: 16.
            in_chans (int): Number of input channels. Default: 1.
            embed_dim (int): Dimension of the embedding. Default: 768.
            depth (int): Number of Transformer blocks. Default: 12.
            num_heads (int): Number of attention heads. Default: 12.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
            norm_layer (nn.Module): Normalization layer to use. Default: nn.LayerNorm.
            mask_ratio (float): Ratio of masked patches. 1 is no masking, 0 is only masked. Default: 0.75.
            pretrained (bool): Whether to load pretrained weights. Default: False.
        """

        super().__init__()

        self.mae = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            mask_ratio=mask_ratio,
        )

        def forward(self, x: Tensor, masks: Tensor) -> TensorDict:
            """
            Forward pass of the model.
            Args:
                x (Tensor): Input tensor of shape (B, C, H, W).
                masks (Tensor): Mask tensor of shape (B, C, H, W).
            Returns:
                TensorDict: A dictionary containing the loss, reconstruction, latent representation, 
                            ids_kept, and ids_restore.
            """

            loss, reconstruction, latent, ids_kept, ids_restore = self.mae(x, masks)

            # Use the CLS token as the output
            latent = latent[:, 0, :]  # (B, D)

            return TensorDict({
                'loss': loss,
                'reconstruction': reconstruction,
                'latent': latent,
                'ids_kept': ids_kept,
                'ids_restore': ids_restore
            }, batch_size=x.shape[:1])
            


class MAESTRAP(nn.Module):
    """ STRAP model for survival analysis.
    
    Args:
        img_size (int): Size of the input image. Default: 224.
        patch_size (int): Size of the patches. Default: 16.
        in_chans (int): Number of input channels. Default: 1.
        embed_dim (int): Dimension of the embedding. Default: 768.
        depth (int): Number of Transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        norm_layer (nn.Module): Normalization layer to use. Default: nn.LayerNorm.
        mask_ratio (float): Ratio of masked patches. Default: 0.75.
        pretrained (bool): Whether to load pretrained weights. Default: False.
    """

    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4,
                 norm_pix_loss: bool = False,
                 threshold: float = 0.5,
                 mask_ratio: float = 0.5,
                 n_tissues: int = 3,
                 n_factors: int = 0,
                 loss_coefficient: float = 0.5
                 ):
        
        super().__init__()

        self.mae = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            depth=depth,
            num_heads=num_heads,
            norm_pix_loss=norm_pix_loss,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            mask_ratio=mask_ratio,
        )

        self.threshold = threshold
        self.n_tissues = n_tissues
        self.n_factors = n_factors
        self.loss_coefficient = loss_coefficient

        self.cox_loss = PartialLikelihood()

        self.fc = nn.Sequential(
            nn.Linear(n_factors, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        self.adapter    = nn.Sequential(
            nn.Linear(2*embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, 1),
        )

    def forward(self,
                images: Tensor,
                masks: Tensor,
                factors: Tensor,
                time: Tensor | None = None,
                event: Tensor | None = None,
                mask_ratio: float = 0.5
                ) -> TensorDict:
        """
        Forward pass of the STRAP model.

        Args:
            images (Tensor): Input tensor of shape (B, C, H, W).
            masks (Tensor): Mask tensor of shape (B, H, W).
            factors (Tensor): Additional risk factors of shape (B, n_factors).
            time (Tensor | None): Survival times. Default: None.
            event (Tensor | None): Event indicators. Default: None.
            mask_ratio (float): Ratio of masked patches. Default: 0.5.
        Returns:
            TensorDict: A dictionary containing the autoencoder loss, Cox loss, total loss, hazards,
                        predictions, factors, time, and event.
        """
        
        # Repeat images at channels dimension
        images = images.repeat(1, masks.shape[1], 1, 1)
        
        # Multiply the masks by the images
        images = images * masks

        # Reshape from (B, C, H, W) to (B*C, H, W)
        images  = images.view(-1, 1, images.shape[2], images.shape[3])
        masks   = masks.view(-1, 1, masks.shape[2], masks.shape[3])

        # Forward pass through the MAE model
        loss, reconstruction, latent, ids_kept, ids_restore = self.mae(images, masks, mask_ratio=mask_ratio)

        # Use the CLS token as the feature representation
        latent = latent[:, 0, :] # (B, embed_dim)


        # If factors are provided, repeat them for each tissue
        if self.n_factors > 0:
            factors = factors.repeat_interleave(self.n_tissues, dim=0).squeeze()
            factors = self.fc(factors) # (B, embed_dim)
            
            # Sum the hazards over the tissues
            fs      = torch.cat([latent, factors], dim=-1) if factors is not None else latent
            hazards = self.adapter(fs)
            hazards = hazards.view(-1, self.n_tissues).mean(dim=1)

            # Compute the Cox partial log likelihood
            if time is not None and event is not None:
                if hazards.shape[0] > 1:
                    hazards = hazards.squeeze()

                cox_loss = self.cox_loss(hazards, time.squeeze(), event.squeeze())
            else:
                cox_loss = torch.tensor(0.0, device=factors.device)
                time = torch.tensor(-1.0, device=factors.device)
                event = torch.tensor(-1.0, device=factors.device)
        else:
            raise ValueError("Factors must be provided for STRAP model.")

        # Get the total loss
        total_loss = loss + self.loss_coefficient*cox_loss

        output = TensorDict({
            "autoencoder_loss": loss,
            "cox_loss": cox_loss,
            "loss": total_loss,
            "hazards": hazards.detach(),
            "predictions": reconstruction.detach(),
            "factors": factors.detach(),
            "time": time.detach(),
            "event": event.detach(),
        })

        return output
    
class STRAP(MAESTRAP):
    
    def forward(self, 
                images: Tensor, 
                masks: Tensor, 
                factors: Tensor, 
                time: Tensor | None = None, 
                event: Tensor | None = None, 
                mask_ratio: float = 0.5) -> TensorDict:
        
        # Repeat images at channels dimension
        images = images.repeat(1, masks.shape[1], 1, 1)
        
        # Multiply the masks by the images
        images = images * masks

        # Reshape from (B, C, H, W) to (B*C, H, W)
        images = images.view(-1, 1, images.shape[2], images.shape[3])
        masks = masks.view(-1, 1, masks.shape[2], masks.shape[3])

        # Pass through the autoencoder
        latent, ids_kept, ids_restore = self.mae.forward_encoder(images, masks, mask_ratio)
        
        # Placeholders for reconstruction and its loss
        reconstruction = torch.tensor(0.0, device=images.device)

        # Use the CLS token as the feature representation
        latent = latent[:, 0, :] # (B, embed_dim)    

        # Repeat the risk factors for each tissue
        if self.n_factors > 0:
            factors = factors.repeat_interleave(self.n_tissues, dim=0).squeeze()

            factors = self.fc(factors) # (B, embed_dim)
        
        fs = torch.cat([latent, factors], dim=-1) if factors is not None else latent
        
        hazards = self.adapter(fs)

        # Sum the hazards over the tissues
        hazards = hazards.view(-1, self.n_tissues).mean(dim=1)

        # Compute the Cox partial log likelihood
        if time is not None and event is not None:
            if hazards.shape[0] > 1:
                hazards = hazards.squeeze()

            # Compute the Cox loss
            cox_loss = self.cox_loss(hazards, time.squeeze(), event.squeeze())
        else:
            cox_loss = torch.tensor(0.0, device=factors.device)
            time = torch.tensor(-1.0, device=factors.device)
            event = torch.tensor(-1.0, device=factors.device)

        # Get the total loss
        total_loss = cox_loss

        output = TensorDict({
            # "autoencoder_loss": loss,
            "cox_loss": cox_loss,
            "loss": total_loss,
            "hazards": hazards.detach(),
            "predictions": reconstruction.detach(),
            "factors": factors.detach(),
            "time": time.detach(),
            "event": event.detach(),
        })

        return output        