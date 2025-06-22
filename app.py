import streamlit as st
import torch
import torch.nn as nn
import numpy as np

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


st.set_page_config(layout="wide", page_title="Conditional GAN Digit Generator")

st.title("Handwritten Digit Generator using cGAN")
st.write("This app uses a Conditional Generative Adversarial Network (cGAN) to generate handwritten digits. Select a digit and click the button to see 5 unique variations.")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = Generator().to(device)
    model.load_state_dict(torch.load('cgan_generator.pth', map_location=device))
    model.eval()
    return model

generator = load_model()
device = torch.device("cpu")

with st.sidebar:
    st.header("Controls")
    selected_digit = st.selectbox(
        "Choose a digit to generate (0-9):",
        options=list(range(10))
    )

    generate_button = st.button("Generate 5 Images", type="primary")

    st.markdown("---")
    st.write("Developed based on a PyTorch cGAN model.")


if generate_button:
    num_variations = 5

    st.subheader(f"Generated {num_variations} Variations for Digit: {selected_digit}")

    with st.spinner(f"Generating {num_variations} images..."):
        noise = torch.randn(num_variations, latent_dim, 1, 1, device=device)
        labels = torch.LongTensor([selected_digit] * num_variations).to(device)

        with torch.no_grad():
            generated_images_tensor = generator(noise, labels)

        generated_images = (generated_images_tensor * 0.5) + 0.5

        cols = st.columns(num_variations)
        for i, col in enumerate(cols):
            with col:
                img_tensor = generated_images[i]
                img_np = img_tensor.cpu().numpy()
                st.image(
                    img_np,
                    caption=f"Variation {i+1}",
                    use_column_width=True
                )
else:
    st.info("Select a digit from the sidebar and click 'Generate 5 Images'.")