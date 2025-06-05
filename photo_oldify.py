import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Css
def set_vintage_ui():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5e7d9;
        }
        .stApp {
            background-image: linear-gradient(to bottom, #f5e7d9, #e8d5c5);
        }
        .sidebar .sidebar-content {
            background-color: #e8d5c5 !important;
            background-image: url('https://www.transparenttextures.com/patterns/cream-paper.png') !important;
            border-right: 2px solid #8b5a2b;
        }
        h1 {
            color: #5c3a21 !important;
            font-family: 'Times New Roman', serif;
            text-shadow: 1px 1px 2px #d4a76a;
        }
        .stButton>button {
            background-color: #8b5a2b !important;
            color: white !important;
            border: 1px solid #5c3a21 !important;
            border-radius: 5px !important;
            font-family: 'Times New Roman', serif;
        }
        .stButton>button:hover {
            background-color: #5c3a21 !important;
            border: 1px solid #8b5a2b !important;
        }
        .stSlider>div>div>div>div {
            background-color: #8b5a2b !important;
        }
        .stCheckbox>label {
            font-family: 'Times New Roman', serif;
            color: #5c3a21 !important;
        }
        .stRadio>label {
            font-family: 'Times New Roman', serif;
            color: #5c3a21 !important;
        }
        .stMarkdown {
            font-family: 'Times New Roman', serif;
            color: #5c3a21 !important;
        }
        .stImage>img {
            border: 8px solid #e8d5c5;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Pengolahan Foto
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
    texture = np.zeros_like(image)
    rows, cols = texture.shape[:2]
    
    for _ in range(int(rows * cols * 0.001)):
        x1, y1 = np.random.randint(0, cols), np.random.randint(0, rows)
        x2, y2 = np.random.randint(0, cols), np.random.randint(0, rows)
        color = np.random.randint(150, 255)
        thickness = np.random.randint(1, 3)
        cv2.line(texture, (x1, y1), (x2, y2), (color, color, color), thickness)
    
    for _ in range(int(rows * cols * 0.0005)):
        x, y = np.random.randint(0, cols), np.random.randint(0, rows)
        radius = np.random.randint(1, 5)
        color = np.random.randint(100, 200)
        cv2.circle(texture, (x, y), radius, (color, color, color), -1)
    
    overlay = cv2.addWeighted(image, 1 - texture_intensity, texture, texture_intensity, 0)
    return overlay

def main():
    # Vintage UI
    set_vintage_ui()
    
    # Header 
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0; border-bottom: 2px solid #8b5a2b;">
            <h1 style="margin: 0;">📷 Vintage Photo Editor</h1>
            <p style="font-family: 'Times New Roman', serif; color: #5c3a21;">
                Transform your photos into timeless vintage masterpieces
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Konten Utama
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### Upload Your Photo")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            st.image(image, caption="Original Image", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### Vintage Effects")
            
            # Kontrol FilterS
            with st.expander("Adjust Vintage Effects", expanded=True):
                use_grayscale = st.checkbox("Grayscale", value=True)
                grayscale_intensity = st.slider("Grayscale Intensity", 0.0, 1.0, 0.7, 0.1) if use_grayscale else 0.0
                
                use_sepia = st.checkbox("Sepia Tone", value=True)
                sepia_intensity = st.slider("Sepia Intensity", 0.0, 1.0, 0.5, 0.1) if use_sepia else 0.0
                
                use_noise = st.checkbox("Film Grain", value=True)
                noise_amount = st.slider("Grain Amount", 0.0, 0.2, 0.05, 0.01) if use_noise else 0.0
                
                use_vignette = st.checkbox("Vignette", value=True)
                vignette_scale = st.slider("Vignette Strength", 0.5, 3.0, 1.5, 0.1) if use_vignette else 0.0
                
                use_blur = st.checkbox("Soft Focus")
                blur_amount = st.slider("Blur Amount", 1, 15, 5, 2) if use_blur else 0
                
                use_texture = st.checkbox("Aging Texture")
                texture_intensity = st.slider("Texture Intensity", 0.0, 1.0, 0.3, 0.1) if use_texture else 0.0
            
            # Proses Foto saat Button di klik
            if st.button("✨ Apply Vintage Effects", use_container_width=True):
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
                
                # Display foto hasil
                st.image(processed_image, caption="Vintage Result", use_container_width=True)
                
                # Button download
                buf = io.BytesIO()
                img_pil = Image.fromarray(processed_image)
                img_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Vintage Photo",
                    data=byte_im,
                    file_name="vintage_photo.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
        else:
            st.markdown(
                """
                <div style="background-color: #e8d5c5; padding: 20px; border-radius: 5px; border: 1px solid #8b5a2b;">
                    <h3 style="color: #5c3a21; text-align: center;">How to use:</h3>
                    <ol style="color: #5c3a21; font-family: 'Times New Roman', serif;">
                        <li>Upload a photo</li>
                        <li>Adjust vintage effects</li>
                        <li>Click "Apply Vintage Effects"</li>
                        <li>Download your masterpiece</li>
                    </ol>
                    <p style="text-align: center; font-style: italic; color: #5c3a21;">
                        "Photography takes an instant out of time, altering life by holding it still."<br>
                        - Dorothea Lange
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; padding: 10px; border-top: 1px solid #8b5a2b;">
            <p style="font-family: 'Times New Roman', serif; color: #5c3a21;">
                © 2025 Vintage Photo Editor | Digital Image Processing | Jenderal Soedirman Univercity
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()