import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import random

def set_vintage_ui():
    st.markdown(
        """
        <style>
        body, div, p, span, label, h1, h2, h3, h4, h5, h6, button {
            color: #5c3a21 !important;
        }
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
        .icon-button {
            font-size: 1.5rem;
            margin: 0 5px;
            cursor: pointer;
        }
        .effect-control {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f0e0d0;
            border-radius: 5px;
            border-left: 3px solid #8b5a2b;
        }
        .tab-container {
            background-color: #f0e0d0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Image processing functions
def apply_grayscale(image, intensity=1.0):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) == 3 and len(gray_image.shape) == 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(image, 1 - intensity, gray_image, intensity, 0)

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
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ubah jadi bilangan ganjil
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

def apply_all_effects(image, effects):
    processed_image = image.copy()
    
    if effects['use_grayscale'] and effects['grayscale_intensity'] > 0:
        processed_image = apply_grayscale(processed_image, effects['grayscale_intensity'])
    
    if effects['use_sepia'] and effects['sepia_intensity'] > 0:
        processed_image = apply_sepia(processed_image, effects['sepia_intensity'])
    
    if effects['use_noise'] and effects['noise_amount'] > 0:
        processed_image = apply_noise(processed_image, effects['noise_amount'])
    
    if effects['use_vignette'] and effects['vignette_scale'] > 0:
        processed_image = apply_vignette(processed_image, effects['vignette_scale'])
    
    if effects['use_blur'] and effects['blur_amount'] > 0:
        processed_image = apply_blur(processed_image, effects['blur_amount'])
    
    if effects['use_texture'] and effects['texture_intensity'] > 0:
        processed_image = apply_texture_overlay(processed_image, effects['texture_intensity'])
    
    return processed_image

def generate_random_effects():
    return {
        'use_grayscale': random.choice([True, False]),
        'grayscale_intensity': random.uniform(0, 1),
        'use_sepia': random.choice([True, False]),
        'sepia_intensity': random.uniform(0, 1),
        'use_noise': random.choice([True, False]),
        'noise_amount': random.uniform(0, 0.2),
        'use_vignette': random.choice([True, False]),
        'vignette_scale': random.uniform(0.5, 3),
        'use_blur': random.choice([True, False]),
        'blur_amount': random.choice([i for i in range(1, 16) if i % 2 == 1]),
        'use_texture': random.choice([True, False]),
        'texture_intensity': random.uniform(0, 1)
    }

# Image processing functions (keep your existing functions)

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
    
    # Initialize session state
    if 'effects' not in st.session_state:
        st.session_state.effects = {
            'use_grayscale': True,
            'grayscale_intensity': 0.7,
            'use_sepia': True,
            'sepia_intensity': 0.5,
            'use_noise': True,
            'noise_amount': 0.05,
            'use_vignette': True,
            'vignette_scale': 1.5,
            'use_blur': False,
            'blur_amount': 5,
            'use_texture': False,
            'texture_intensity': 0.3
        }
    
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload or Capture Photo")
        
        # Tab interface for upload vs camera
        tab1, tab2 = st.tabs(["📁 Upload Photo", "📷 Take Photo"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                try:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.session_state.original_image = image
                    st.image(image, caption="Original Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        
        with tab2:
            st.markdown("Use your camera to take a photo")
            picture = None
            try:
                picture = st.camera_input("Take a picture", label_visibility="collapsed")
            except Exception as e:
                st.warning(f"Camera error: {str(e)}")
                st.info("Please allow camera access or try uploading a file instead")
            
            if picture is not None:
                try:
                    file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.session_state.original_image = image
                    st.image(image, caption="Captured Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing captured image: {str(e)}")
        
        # Show preview in the same column (below the original/captured image)
        if st.session_state.original_image is not None:
            st.markdown("### 🖼️  Preview")
            try:
                processed_image = apply_all_effects(st.session_state.original_image, st.session_state.effects)
                st.image(processed_image, caption="Processed Image", use_container_width=True)
                
                
                # Random effects button
                if st.button("🎲 Random Vintage Preset", use_container_width=True):
                    st.session_state.effects = generate_random_effects()
                    
                # Download button moved here
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
            except Exception as e:
                st.error(f"Error applying effects: {str(e)}")
    
    with col2:
        if st.session_state.original_image is not None:
            st.markdown("### 🎨 Vintage Effects Controls")
            
            
            # Effect controls
            with st.expander("Adjust Effects", expanded=True):
                st.markdown('<div class="effect-control">'
                           '🌑 <span style="margin-left: 10px;">Grayscale</span>'
                           '</div>', unsafe_allow_html=True)
                st.session_state.effects['use_grayscale'] = st.checkbox(
                    "Enable Grayscale", 
                    value=st.session_state.effects['use_grayscale'],
                    key="grayscale_check"
                )
                if st.session_state.effects['use_grayscale']:
                    st.session_state.effects['grayscale_intensity'] = st.slider(
                        "Grayscale Intensity", 
                        0.0, 1.0, 
                        value=st.session_state.effects['grayscale_intensity'], 
                        step=0.1,
                        key="grayscale_slider"
                    )
                
                st.markdown('<div class="effect-control">'
                           '🟤 <span style="margin-left: 10px;">Sepia Tone</span>'
                           '</div>', unsafe_allow_html=True)
                st.session_state.effects['use_sepia'] = st.checkbox(
                    "Enable Sepia Tone", 
                    value=st.session_state.effects['use_sepia'],
                    key="sepia_check"
                )
                if st.session_state.effects['use_sepia']:
                    st.session_state.effects['sepia_intensity'] = st.slider(
                        "Sepia Intensity", 
                        0.0, 1.0, 
                        value=st.session_state.effects['sepia_intensity'], 
                        step=0.1,
                        key="sepia_slider"
                    )
                
                st.markdown('<div class="effect-control">'
                           '🎞️ <span style="margin-left: 10px;">Film Grain</span>'
                           '</div>', unsafe_allow_html=True)
                st.session_state.effects['use_noise'] = st.checkbox(
                    "Enable Film Grain", 
                    value=st.session_state.effects['use_noise'],
                    key="noise_check"
                )
                if st.session_state.effects['use_noise']:
                    st.session_state.effects['noise_amount'] = st.slider(
                        "Grain Amount", 
                        0.0, 0.2, 
                        value=st.session_state.effects['noise_amount'], 
                        step=0.01,
                        key="noise_slider"
                    )
                
                st.markdown('<div class="effect-control">'
                           '⭕ <span style="margin-left: 10px;">Vignette</span>'
                           '</div>', unsafe_allow_html=True)
                st.session_state.effects['use_vignette'] = st.checkbox(
                    "Enable Vignette", 
                    value=st.session_state.effects['use_vignette'],
                    key="vignette_check"
                )
                if st.session_state.effects['use_vignette']:
                    st.session_state.effects['vignette_scale'] = st.slider(
                        "Vignette Strength", 
                        0.5, 3.0, 
                        value=st.session_state.effects['vignette_scale'], 
                        step=0.1,
                        key="vignette_slider"
                    )
                
                st.markdown('<div class="effect-control">'
                           '🔍 <span style="margin-left: 10px;">Soft Focus</span>'
                           '</div>', unsafe_allow_html=True)
                st.session_state.effects['use_blur'] = st.checkbox(
                    "Enable Soft Focus", 
                    value=st.session_state.effects['use_blur'],
                    key="blur_check"
                )
                if st.session_state.effects['use_blur']:
                    st.session_state.effects['blur_amount'] = st.slider(
                        "Blur Amount", 
                        1, 15, 
                        value=st.session_state.effects['blur_amount'], 
                        step=2,
                        key="blur_slider"
                    )
                
                st.markdown('<div class="effect-control">'
                           '🧱 <span style="margin-left: 10px;">Aging Texture</span>'
                           '</div>', unsafe_allow_html=True)
                st.session_state.effects['use_texture'] = st.checkbox(
                    "Enable Aging Texture", 
                    value=st.session_state.effects['use_texture'],
                    key="texture_check"
                )
                if st.session_state.effects['use_texture']:
                    st.session_state.effects['texture_intensity'] = st.slider(
                        "Texture Intensity", 
                        0.0, 1.0, 
                        value=st.session_state.effects['texture_intensity'], 
                        step=0.1,
                        key="texture_slider"
                    )
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="background-color: #e8d5c5; padding: 20px; border-radius: 5px; border: 1px solid #8b5a2b;">
                    <h3 style="color: #5c3a21; text-align: center;">How to use:</h3>
                    <ol style="color: #5c3a21; font-family: 'Times New Roman', serif;">
                        <li>Upload a photo or take one with your camera</li>
                        <li>Adjust vintage effects (or click "Random Preset")</li>
                        <li>See preview of your changes below the original</li>
                        <li>Download your masterpiece</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; padding: 10px; border-top: 1px solid #8b5a2b;">
            <p style="font-family: 'Times New Roman', serif; color: #5c3a21;">
                © 2025 Vintage Photo Editor | Digital Image Processing | Jenderal Soedirman University
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()