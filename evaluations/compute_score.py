import numpy as np
import cv2
from cleanfid import fid
from cleanfid import prdc

# score_clean = fid.compute_fid("folder_real", "folder_fake", mode="clean", num_workers=0)
# print(f"clean-fid score is {score_clean:.3f}")

folder_real = "/data/lh/docker/dataset/HieraFashion_5K/test"
folder_fake = "/data/lh/docker/project/TexFit/outputs/texfit_offcial_A2"

score_clip_fid = fid.compute_fid(folder_real, folder_fake, mode="clean", model_name="clip_vit_b_32")
print(f"clip-fid score is {score_clip_fid:.3f}")


# score_clip_prdc = prdc.prdc_scores(folder_real, folder_fake, mode="clean", model_name="clip_vit_b_32")
# print(f"clip-prdc score is {score_clip_prdc}")