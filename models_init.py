from segment_anything import build_sam, SamPredictor
import torch.cuda
import random

if torch.cuda.is_available():
    SUPIR_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

def initialize_models():
    # Load SAM
    SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert SAM_CHECKPOINT, 'SAM_CHECKPOINT is not found!'
    sam = build_sam(checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    model = SamPredictor(sam)
    return model