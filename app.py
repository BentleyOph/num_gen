import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid


latent_dim = 100
num_classes = 10
embedding_dim = 100
channels_img = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embed = nn.Embedding(num_classes, embedding_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embedding_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        lbl_embedding = self.embed(labels).view(-1, self.embed.embedding_dim, 1, 1)
        gen_input = torch.cat((noise, lbl_embedding), dim=1)
        return self.net(gen_input)

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="Conditional GAN Digit Generator")

st.title("Handwritten Digit Generator using cGAN")
st.write("This app uses a Conditional Generative Adversarial Network (cGAN) trained on the MNIST dataset to generate handwritten digits. Select a digit and click 'Generate'!")

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    # We run this on CPU as it's fast enough for inference
    device = torch.device("cpu")
    model = Generator().to(device)
    # Load the trained weights
    model.load_state_dict(torch.load('cgan_generator.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

generator = load_model()
device = torch.device("cpu")

# --- UI Elements ---
with st.sidebar:
    st.header("Controls")
    selected_digit = st.selectbox(
        "Choose a digit to generate (0-9):",
        options=list(range(10))
    )

    generate_button = st.button("Generate Image", type="primary")

    st.markdown("---")
    st.write("Developed based on a PyTorch cGAN model.")


# --- Generation Logic ---
if generate_button:
    st.subheader(f"Generated Image for Digit: {selected_digit}")

    with st.spinner("Generating..."):
        # 1. Prepare inputs for the generator
        # Create the noise vector
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        # Create the label tensor for the selected digit
        label = torch.LongTensor([selected_digit]).to(device)

        # 2. Generate the image
        with torch.no_grad():
            generated_image_tensor = generator(noise, label)

        # 3. Post-process the tensor to display it
        # The generator's output is in the range [-1, 1], so we need to normalize it to [0, 1] for display
        generated_image = (generated_image_tensor * 0.5) + 0.5
        # Squeeze the batch dimension and convert to numpy for display
        img_np = generated_image.squeeze().cpu().numpy()

        # Display the image
        st.image(img_np, caption=f"Generated Digit: {selected_digit}", width=256)

else:
    st.info("Select a digit from the sidebar and click 'Generate'.")