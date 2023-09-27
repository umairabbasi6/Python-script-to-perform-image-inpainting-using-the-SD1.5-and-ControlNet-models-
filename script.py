import torch
import numpy as np
from PIL import Image

class ImageInpainting:
    def __init__(self, sd15_model, controlnet_model):
        self.sd15_model = sd15_model
        self.controlnet_model = controlnet_model

    def inpaint(self, image, mask):
        """
        Inpaint the masked area in the image.

        Args:
            image: A PIL Image object.
            mask: A PIL Image object with the same size as the image, with 0 pixels indicating the masked area.

        Returns:
            A PIL Image object with the inpainted masked area.
        """

        # Convert the image and mask to PyTorch tensors.
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        mask_tensor = torch.from_numpy(np.array(mask)).float()

        # Generate a noise image.
        noise_image = torch.randn(image_tensor.size(), device=image_tensor.device)

        # Inpaint the noise image using the SD1.5 model.
        inpainted_noise_image = self.sd15_model(noise_image, mask_tensor)

        # Generate a control image using the ControlNet model.
        control_image = self.controlnet_model(inpainted_noise_image)

        # Combine the inpainted noise image and the control image to generate the final image.
        final_image = (inpainted_noise_image * (1 - mask_tensor)) + (image_tensor * mask_tensor) + control_image

        # Convert the final image to a PIL Image object.
        final_image = Image.fromarray((final_image.cpu().numpy() * 255.0).astype(np.uint8))

        return final_image

if __name__ == '__main__':
    # Load the SD1.5 and ControlNet models.
    sd15_model = torch.load('yoursd1.5 model')
    controlnet_model = torch.load('your controlnet model')

    # Create an ImageInpainting object.
    image_inpainting = ImageInpainting(sd15_model, controlnet_model)

    # Load the trained image and mask.
    image = Image.open('ypur_image.png')
    mask = Image.open('Your_trained_mask.png')

    # Inpaint the image.
    inpainted_image = image_inpainting.inpaint(image, mask)

    # Save the inpainted image.
    inpainted_image.save('inpainted_image.png')
