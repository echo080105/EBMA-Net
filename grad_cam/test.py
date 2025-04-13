import torch
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from nets.unet import Unet


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to get gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)

        # Zero gradients
        self.model.zero_grad()

        # Get the score for the target class
        target = output[0, target_class]

        # Backward pass
        target.backward()

        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Compute the weight for each feature map
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Compute the weighted sum of the feature maps
        cam = torch.sum(weights * activations, dim=1)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize the CAM
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def apply_heatmap(cam, original_image):
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlay_image = heatmap + np.float32(original_image) / 255
    overlay_image = overlay_image / np.max(overlay_image)
    return np.uint8(255 * overlay_image)


model = Unet(num_classes=4)
# Example of using GradCAM
# Assume 'model' is your CMSAF-Net model and 'input_image' is your test image
target_layer = model.out_conv  # Choose the last conv layer or the one close to the output
grad_cam = GradCAM(model, target_layer)

# Forward pass with the input image
input_image = ...  # Your preprocessed input image here
target_class = ...  # The target class you are interested in

# Generate CAM
cam = grad_cam.generate_cam(input_image.unsqueeze(0), target_class)

# Convert to numpy array and apply heatmap
original_image = cv2.imread("./Rust_278.jpg")  # Load your original image for overlay
overlay_image = apply_heatmap(cam, original_image)

# Save or display the image
cv2.imwrite("cam_overlay.png", overlay_image)
cv2.imshow("Grad-CAM", overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
