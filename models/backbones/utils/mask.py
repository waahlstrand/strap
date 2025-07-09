import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch
from torch import Tensor
from typing import Tuple, List

class MaskEmbed(nn.Module):
    r""" Mask to patch token embedding.

    Creates mask tokens from an input mask. 
    The mask is divided into patches and the mean value of each patch is used
    as a cutoff to determine if the patch is masked or not.

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.

    Returns:
        Tensor: Returns the mask tokens (B, H, W)        
    """

    def __init__(self, img_size: int = 224, patch_size: int= 4, threshold: float = 0.5):

        super().__init__()

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.threshold = threshold

        self.proj = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, mask: Tensor) -> Tensor:

        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        B, _, H, W = mask.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}, {mask.shape}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        patches: Tensor = self.proj(mask) # B, patch_size * patch_size, num_patches


        tokens  = patches.mean(dim=1) > self.threshold
        tokens  = tokens.float().view(B, *self.patches_resolution)

        return tokens
    

class MaskEmbed3D(nn.Module):
    r""" Mask to patch token embedding.

    Creates mask tokens from an input mask. 
    The mask is divided into patches and the mean value of each patch is used
    as a cutoff to determine if the patch is masked or not.

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.

    Returns:
        Tensor: Returns the mask tokens (B, H, W)        
    """

    def __init__(self, 
                 img_size: Tuple[int, int, int] = (110, 512, 512),
                 patch_size: Tuple[int, int, int] = (11, 16, 16),
                 threshold: float = 0.5):

        super().__init__()
        
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]
        self.threshold = threshold

        kernel_size = list(patch_size)
        self.proj = lambda x: x.unfold(2, kernel_size[0], kernel_size[0])\
                               .unfold(3, kernel_size[1], kernel_size[1])\
                               .unfold(4, kernel_size[2], kernel_size[2])

    def forward(self, mask: Tensor) -> Tensor:

        if len(mask.shape) == 4:
            # Ensure the mask has a channel dimension
            mask = mask.unsqueeze(1)

        B, C, T, H, W = mask.shape
        assert T == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({T}*{H}*{W}, {mask.shape}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        patches: Tensor = self.proj(mask) # B, t_patch_size, patch_size patch_size, num_patches_t, num_patches_h, num_patches_w

        tokens  = patches.float().mean(dim=(-1, -2, -3)) > self.threshold
        tokens  = tokens.float().view(B, -1, 1) # B, num_patches, 1

        return tokens
    

    
def remove_empty_patches_and_pad(xs: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
	B, L, D = xs.shape
	B, P, _ = masks.shape
	
	# assert L == P ** 2, f"Number of patches in image tensor ({L}) doesn't match mask tensor ({P}x{P})."
    # Get indices of patches with all zeros
    # this will be a variable number of patches
    # for each image in the batch

    # Contains the indices of the non-empty patches in the shape (N, 1)
	masks = masks.view(B, -1)
	
	mask_list = []
	x_list = []
	lengths = []
	idxs = []
	
	# Use torch gather to get the non-empty patches
	for i in range(B):
		mask = masks[i]
		x = xs[i]
		non_empty_idx = torch.nonzero(mask == 1, as_tuple=True)

		mask = mask[non_empty_idx[0]]
		x = x[non_empty_idx[0]]
        
		mask_list.append(mask)
		x_list.append(x)
		lengths.append(len(non_empty_idx[0]))
		idxs.append(non_empty_idx[0])
        
	# Pad the sequences
	xs = pad_sequence(x_list, batch_first=True)
	masks = pad_sequence(mask_list, batch_first=True)
	lengths = torch.tensor(lengths)
	# idxs = torch.tensor(idxs)
	
	return xs, masks, lengths, idxs
    

def restore_patch_sequence_to_image(
        patches: Tensor, 
        lengths: Tensor, 
        idxs: Tensor,
        n_patches: int) -> Tensor:
    r"""
    Restore the patches to the original image.

    Args:
        patches (Tensor): Patches tensor (B, L, D)
        lengths (Tensor): Lengths of each sample in the batch (B)
        non_empty_idxs (Tensor): Indices of non-empty patches (N, 1)

    Returns:
        Tensor: Returns the restored image tensor (B, H, W, D)
    """    
    B, L, D = patches.shape
    restored = torch.zeros(B, n_patches*n_patches, D, device=patches.device)
    for sample_idx in range(B):

        # Get all tokens for the sample, excluding the padding
        sequence = patches[sample_idx, :lengths[sample_idx]]

        # Get their coordinates in the original image
        coords = idxs[sample_idx]

        # Restore the image
        print(sequence.shape, len(coords))
        restored[sample_idx, coords] = sequence

    restored = restored.view(B, n_patches, n_patches)

    return restored 

def ids_to_mask(idxs: Tensor, n_patches: int) -> Tensor:
    r"""
    Convert the indices of the non-empty patches to a mask tensor.

    Args:
        idxs (Tensor): Indices of the non-empty patches (N, 1)
        n_patches (int): Number of patches in the image

    Returns:
        Tensor: Returns the mask tensor (1, H, W)
    """
    mask = torch.zeros(n_patches, dtype=torch.int8)
    mask[idxs] = 1
    mask = mask.view(1, -1)
    return mask

def random_masking(
          patches: Tensor, 
          lengths: Tensor, 
          patch_idxs: List[Tensor], 
          mask_ratio: float = 0.5
          ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
    """
    Perform per-sample random-masking of the patch sequence. The patch sequences x
    are padded with original length given by 'lengths', and the indices of the non-empty
    patches are given by 'idxs'.
     
    Args:
        x (Tensor): The patch sequence tensor (B, L, D)
        lengths (Tensor): The lengths of the original sequences (B)
        idxs (List[Tensor]): The indices of the non-empty patches (B, N)
        mask_ratio (float): The ratio of patches to mask. Default: 0.5
         
    Returns:
        Tuple[Tensor, List[Tensor], List[Tensor]]: 
            - masked (Tensor): The masked patch sequences (B, L, D)
            - masked_idxs (List[Tensor]): The indices of the masked patches (B, M)
            - to_restore_idxs (List[Tensor]): The indices of the patches to restore (B, M)
    """

    B, L, D = patches.shape
    masked = []
    masked_idxs = []
    to_restore_idxs = []
    new_lengths = []
    for i in range(B):
        
        length      = int(lengths[i].item())
        patch       = patches[i][:length] # Remove padding
        patch_idx   = patch_idxs[i]
        
        # Get random indices to mask
        mask_length = int(length * mask_ratio)
        random_permutation = torch.randperm(length)
        
        mask_idx    = random_permutation[:mask_length]
        to_restore = random_permutation[mask_length:]

        # Mask the patches
        masked_patch_idxs   = patch_idx[mask_idx]
        masked_patch_to_restore_idxs = patch_idx[to_restore]
        masked_patch        = patch[mask_idx]

        masked.append(masked_patch)
        masked_idxs.append(masked_patch_idxs)
        to_restore_idxs.append(masked_patch_to_restore_idxs)
        new_lengths.append(mask_length)

    masked = pad_sequence(masked, batch_first=True)
    new_lengths = torch.tensor(new_lengths)


    return masked, masked_idxs, to_restore_idxs
