import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_sepia(image, intensity=0.5):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return cv2.addWeighted(image, 1 - intensity, sepia_image, intensity, 0)

def apply_noise(image, amount=0.05):
    noise = np.random.normal(0, amount * 255, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_vignette(image, vignette_scale=1.5):
    rows, cols = image.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, vignette_scale * cols)
    kernel_y = cv2.getGaussianKernel(rows, vignette_scale * rows)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    
    if len(image.shape) == 3:
        mask = np.dstack([mask] * 3)
    
    vignette_image = image * mask
    return vignette_image.astype(np.uint8)

def apply_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_texture_overlay(image, texture_intensity=0.3):
    # Create a scratch texture
    texture = np.zeros_like(image)
    rows, cols = texture.shape[:2]
    
    # Add random scratches
    for _ in range(int(rows * cols * 0.001)):
        x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
        x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
        color = np.random.randint(150, 255)
        thickness = np.random.randint(1, 3)
        cv2.line(texture, (x1, y1), (x2, y2), (color, color, color), thickness)
    
    # Add random dust spots
    for _ in range(int(rows * cols * 0.0005)):
        x, y = np.random.randint(0, cols), np.random.randint(0, rows)
        radius = np.random.randint(1, 5)
        color = np.random.randint(100, 200)
        cv2.circle(texture, (x, y), radius, (color, color, color), -1)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - texture_intensity, texture, texture_intensity, 0)
    return overlay

def main():
    st.title("Photo Oldify - Aging Filter")
    st.write("Upload a photo to apply vintage effects")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Sidebar controls
        st.sidebar.header("Filter Controls")
        
        # Grayscale
        use_grayscale = st.sidebar.checkbox("Grayscale")
        grayscale_intensity = st.sidebar.slider("Grayscale Intensity", 0.0, 1.0, 1.0, 0.1) if use_grayscale else 0.0
        
        # Sepia
        use_sepia = st.sidebar.checkbox("Sepia")
        sepia_intensity = st.sidebar.slider("Sepia Intensity", 0.0, 1.0, 0.5, 0.1) if use_sepia else 0.0
        
        # Noise
        use_noise = st.sidebar.checkbox("Noise")
        noise_amount = st.sidebar.slider("Noise Amount", 0.0, 0.2, 0.05, 0.01) if use_noise else 0.0
        
        # Vignette
        use_vignette = st.sidebar.checkbox("Vignette")
        vignette_scale = st.sidebar.slider("Vignette Strength", 0.5, 3.0, 1.5, 0.1) if use_vignette else 0.0
        
        # Blur
        use_blur = st.sidebar.checkbox("Blur")
        blur_amount = st.sidebar.slider("Blur Amount", 1, 15, 5, 2) if use_blur else 0
        
        # Texture Overlay
        use_texture = st.sidebar.checkbox("Texture Overlay")
        texture_intensity = st.sidebar.slider("Texture Intensity", 0.0, 1.0, 0.3, 0.1) if use_texture else 0.0
        
        # Apply filters
        processed_image = image.copy()
        
        if use_grayscale and grayscale_intensity > 0:
            gray_image = apply_grayscale(processed_image)
            if len(processed_image.shape) == 3 and len(gray_image.shape) == 2:
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
            processed_image = cv2.addWeighted(processed_image, 1 - grayscale_intensity, gray_image, grayscale_intensity, 0)
        
        if use_sepia and sepia_intensity > 0:
            processed_image = apply_sepia(processed_image, sepia_intensity)
        
        if use_noise and noise_amount > 0:
            processed_image = apply_noise(processed_image, noise_amount)
        
        if use_vignette and vignette_scale > 0:
            processed_image = apply_vignette(processed_image, vignette_scale)
        
        if use_blur and blur_amount > 0:
            processed_image = apply_blur(processed_image, blur_amount)
        
        if use_texture and texture_intensity > 0:
            processed_image = apply_texture_overlay(processed_image, texture_intensity)
        
        # Display processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        
        # Download button
        buf = io.BytesIO()
        img_pil = Image.fromarray(processed_image)
        img_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Processed Image",
            data=byte_im,
            file_name="oldified_photo.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()