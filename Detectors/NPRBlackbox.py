import PIL.Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
from Detectors.NPRBase import *
from collections import OrderedDict
from copy import deepcopy
import random
import os
from Detectors.NPRBase import get_model
from Config import device


class BlackBoxExplainer:
    def __init__(self, model, device=device, input_size=512):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.input_size = input_size
        self.class_names = ["Real", "Fake"]

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def generate_masks(self, N, h, w, p):
        small_masks = (torch.rand(N, 1, h, w) < p).float()
        upsampled = torch.nn.functional.interpolate(
            small_masks,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False
        )
        return upsampled

    def explain(self, image: PIL.Image.Image, target_class=None, N=5000, h=4, w=4, p=0.3, batch_size=128, seed=42):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.sigmoid(output).item()
            pred_class = 1 if probs > 0.5 else 0
            if pred_class == 0:
                confidence = 1 - probs
            else:
                confidence = probs

        if target_class is None:
            target_class = pred_class

        masks = self.generate_masks(N, h, w, p).to(self.device)
        img_batch = img_tensor.repeat(N, 1, 1, 1)
        #masked_images = img_batch * masks

        all_scores = []
        for i in tqdm(range(0, N, batch_size), desc="Processing masks"):
            batch_masks = masks[i:i + batch_size].to(self.device)
            batch_imgs = img_tensor.repeat(batch_masks.size(0), 1, 1, 1)
            masked_batch = batch_imgs * batch_masks

            with torch.no_grad():
                batch_output = self.model(masked_batch)
                batch_probs = torch.sigmoid(batch_output).squeeze(1)
                batch_scores = batch_probs if target_class == 1 else 1 - batch_probs

            all_scores.append(batch_scores.cpu())  # Move to CPU

            del batch_masks, batch_imgs, masked_batch, batch_output, batch_probs, batch_scores
            torch.cuda.empty_cache()

        scores = torch.cat(all_scores)
        saliency_map = torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32, device='cpu')
        for i in tqdm(range(0, N, batch_size), desc="Accumulating saliency"):
            batch_masks = masks[i:i + batch_size].cpu()
            batch_scores = scores[i:i + batch_size].view(-1, 1, 1, 1)  # already on CPU
            saliency_map += torch.sum(batch_masks * batch_scores, dim=0)

        saliency = saliency_map.squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)

        def more_normalization(saliency_input):

            # Normalize the saliency map using percentile clipping (top 1% and bottom 1%)
            low, high = np.percentile(saliency_input, [10, 90])
            saliency_output = (saliency_input - low) / (high - low + 1e-8)
            saliency_output = np.clip(saliency_output, 0, 1)  # Ensure values are in range [0, 1]

            # Apply scaling to make the mean value closer to 0.5
            mean_value = np.mean(saliency_output)
            scaling_factor = 0.5 / (mean_value + 1e-8)
            saliency_output *= scaling_factor

            # Threshold to suppress less important regions (set values below 0.1 to zero)
            saliency_output = np.where(saliency_output > 0.1, saliency_output, 0)  # Example threshold

            # Apply Gaussian blur to smooth out the map
            saliency_output = cv2.GaussianBlur(saliency_output, (11, 11), sigmaX=5)

            return saliency_output

        return {
            '-': saliency,
            'saliency_norm': more_normalization(saliency),
            'target_class': target_class,
            'class_name': self.class_names[target_class],
            'confidence': confidence,
        }



def visualize_explanation(image: PIL.Image.Image, explanation):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    title = (f"Original Image\n"
             f"Predicted: {explanation['class_name']}\n"
             f"Confidence: {explanation['confidence']:.2%}")
    ax1.imshow(image)
    ax1.set_title(title)
    ax1.axis('off')

    ax2.imshow(explanation['saliency_map'], cmap='hot')
    ax2.set_title("Saliency")
    ax2.axis('off')

    overlay_alpha = 0.7
    img_resized = image.resize((512, 512))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    cmap = plt.get_cmap('jet')
    heatmap = cmap(explanation['saliency_map'])[:, :, :3]
    overlay = (1 - overlay_alpha) * img_np + overlay_alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    ax3.imshow(overlay)
    ax3.set_title("Boosted Heatmap Overlay")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def blackbox(input_pil: PIL.Image.Image):
    explanation = explainer.explain(input_pil)
    return explanation


def get_top_saliency(saliency: np.ndarray, threshold=80):
    threshold = np.percentile(saliency, threshold)

    mask = np.zeros_like(saliency, dtype=np.uint8)
    mask[saliency >= threshold] = 255
    return mask
explainer = BlackBoxExplainer(get_model(), device=device, input_size=224)


