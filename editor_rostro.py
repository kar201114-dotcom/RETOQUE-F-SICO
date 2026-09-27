import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Estudio de Retoque Web", layout="wide")

st.title("🎨 Estudio de Retoque de Imagen en la Nube")
st.write("Sube una foto y ajusta el brillo, contraste y los tonos de piel usando los controles laterales.")

# --- Panel Lateral (Controles Web) ---
st.sidebar.header("Herramientas de Retoque")
archivo_subido = st.sidebar.file_uploader("Sube tu imagen (Rostro/Cuerpo)", type=["jpg", "png", "jpeg"])

# Barritas deslizantes web (equivalentes a los trackbars)
contraste = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
brillo = st.sidebar.slider("Brillo", -100, 100, 0)
tono_piel = st.sidebar.slider("Tono de Piel (Matiz)", 0, 180, 0)

if archivo_subido is not None:
    # Leer la imagen subida con OpenCV
    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Aplicar contraste y brillo
    img_ajustada = cv2.convertScaleAbs(img_original, alpha=contraste, beta=brillo)
    
    # Modificador inteligente de tono de piel si se ajusta la barra
    if tono_piel > 0:
        hsv = cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2HSV)
        # Rango estimado para tonos de piel en HSV
        mask_piel = cv2.inRange(hsv, (0, 20, 70), (25, 250, 255))
        # Desplazar el matiz en la zona de piel detectada
        hsv[:, :, 0] = np.where(mask_piel > 0, (hsv[:, :, 0].astype(int) + tono_piel) % 180, hsv[:, :, 0])
        img_ajustada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    # Mostrar las imágenes lado a lado en la web
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        # Streamlit usa RGB, OpenCV usa BGR, por eso convertimos los colores para mostrarlas bien
        st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("Imagen Retocada")
        st.image(cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    # Opcional: Permitir descargar la imagen resultante
    resultado_pil = Image.fromarray(cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2RGB))
    
    # Guardar temporalmente para descarga
    resultado_pil.save("resultado_retocado.jpg")
    with open("resultado_retocado.jpg", "rb") as file:
        st.sidebar.download_button(
            label="📥 Descargar Imagen Retocada",
            data=file,
            file_name="imagen_retocada.jpg",
            mime="image/jpeg"
        )
else:
    st.info("👈 Por favor, sube una imagen en el panel lateral para comenzar a editar.")
