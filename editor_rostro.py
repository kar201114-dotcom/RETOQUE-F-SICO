# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 00:24:46 2026

@author: karma
"""

import cv2
import numpy as np

# Variables globales para el manejo del mouse y deformación
dragging = False
pt_start = (0, 0)
pt_end = (0, 0)
imagen_trabajo = None
imagen_original = None

def mouse_liquify(event, x, y, flags, param):
    global dragging, pt_start, pt_end, imagen_trabajo
    
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        pt_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        pt_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        pt_end = (x, y)
        # Al soltar el mouse, aplicamos la deformación local (efecto empujar/estirar)
        imagen_trabajo = aplicar_deformacion_local(imagen_trabajo, pt_start, pt_end, radio=40, fuerza=15)
        actualizar_pantalla()

def aplicar_deformacion_local(img, p1, p2, radio=40, fuerza=15):
    """
    Simula un efecto Liquify local: deforma los píxeles alrededor de p1 moviéndolos hacia p2.
    Ideal para afilar rostro, encoger nariz, agrandar labios o dar volumen.
    """
    h, w = img.shape[:2]
    resultado = img.copy()
    
    x1, y1 = p1
    x2, y2 = p2
    
    # Vector de desplazamiento
    dx = x2 - x1
    dy = y2 - y1
    distancia_mov = np.sqrt(dx**2 + dy**2)
    if distancia_mov < 1:
        return img
        
    # Definir la zona de influencia basada en el radio
    x_min = max(0, int(min(x1, x2) - radio))
    x_max = min(w, int(max(x1, x2) + radio))
    y_min = max(0, int(min(y1, y2) - radio))
    y_max = min(h, int(max(y1, y2) + radio))
    
    # Crear mallas de coordenadas locales
    grid_y, grid_x = np.mgrid[y_min:y_max, x_min:x_max]
    
    # Calcular distancia de cada píxel de la zona al punto inicial del arrastre
    distancias = np.sqrt((grid_x - x1)**2 + (grid_y - y1)**2)
    
    # Máscara de influencia gaussiana suave
    mascara = np.exp(-(distancias**2) / (2 * (radio**2)**0.5))
    mascara[distancias > radio] = 0
    
    # Desplazamiento calculado para cada píxel afectado
    map_x = (grid_x - dx * mascara * (fuerza / radio)).astype(np.float32)
    map_y = (grid_y - dy * mascara * (fuerza / radio)).astype(np.float32)
    
    # Recortar la región de interés y aplicar remapeo local
    roi = img[y_min:y_max, x_min:x_max]
    roi_deformada = cv2.remap(roi, map_x - x_min, map_y - y_min, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    resultado[y_min:y_max, x_min:x_max] = roi_deformada
    return resultado

def actualizar_pantalla(val=None):
    """Aplica los cambios de las barritas (contraste, brillo, tono de piel) y muestra la imagen."""
    global imagen_trabajo, imagen_original
    
    if imagen_original is None:
        return
        
    # Leer valores de las barritas deslizantes
    contraste = cv2.getTrackbarPos('Contraste (x0.1)', 'Estudio de Retoque') / 10.0 # Rango 0.5 a 2.0 (por defecto 10 -> 1.0)
    brillo = cv2.getTrackbarPos('Brillo (-100 a 100)', 'Estudio de Retoque') - 100 # Rango -100 a 100
    tono_piel = cv2.getTrackbarPos('Tono de Piel (Matiz)', 'Estudio de Retoque') # Rango 0 a 180
    
    # Aplicar contraste y brillo sobre la imagen ya deformada
    img_ajustada = cv2.convertScaleAbs(imagen_trabajo, alpha=contraste, beta=brillo)
    
    # Modificador inteligente de tono de piel (si se ajusta la barrita de matiz)
    if tono_piel > 0:
        hsv = cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2HSV)
        # Rango estimado para tonos de piel en HSV
        mask_piel = cv2.inRange(hsv, (0, 20, 70), (25, 250, 255))
        # Desplazar el matiz en la zona de piel detectada
        hsv[:, :, 0] = np.where(mask_piel > 0, (hsv[:, :, 0].astype(int) + tono_piel) % 180, hsv[:, :, 0])
        img_ajustada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    cv2.imshow('Estudio de Retoque', img_ajustada)

# --- PROGRAMA PRINCIPAL ---
if __name__ == "__main__":
    print("=== ESTUDIO DE RETOCAJE Y LUZ INTERACTIVO ===")
    ruta = input("Ingresa la ruta de la imagen con el rostro/cuerpo: ").strip().replace('"', '')
    
    imagen_original = cv2.imread(ruta)
    if imagen_original is None:
        print("[Error] No se pudo cargar la imagen. Revisa la ruta.")
    else:
        # Redimensionar si es muy grande para que quepa bien en pantalla
        h_orig, w_orig = imagen_original.shape[:2]
        if h_orig > 800 or w_orig > 1000:
            imagen_original = cv2.resize(imagen_original, (w_orig // 2, h_orig // 2))
            
        imagen_trabajo = imagen_original.copy()
        
        # Crear ventana y asignar callbacks del mouse
        cv2.namedWindow('Estudio de Retoque')
        cv2.setMouseCallback('Estudio de Retoque', mouse_liquify)
        
        # Crear barritas deslizantes (Trackbars) que van y vienen de forma fluida
        # Nota: El contraste por defecto estará en 10 (que representa 1.0)
        cv2.createTrackbar('Contraste (x0.1)', 'Estudio de Retoque', 10, 20, actualizar_pantalla)
        # Brillo con un valor base de 100 para que represente 0 (rango de 0 a 200 -> -100 a +100)
        cv2.createTrackbar('Brillo (-100 a 100)', 'Estudio de Retoque', 100, 200, actualizar_pantalla)
        # Tono de piel (0 significa sin alterar)
        cv2.createTrackbar('Tono de Piel (Matiz)', 'Estudio de Retoque', 0, 180, actualizar_pantalla)
        
        print("\n--- INSTRUCCIONES DE USO ---")
        print("1. **DEFORMACIÓN (Liquify):** Haz clic con el botón izquierdo sobre la zona que quieras modificar (ej. nariz, labios, contorno de mandíbula, o volumen corporal) y **arrastra el mouse** hacia donde quieras estirar o empujar la carne/facciones.")
        print("2. **BARRITAS DESLIZANTES:** Mueve los controles deslizantes arriba de la ventana en tiempo real para ajustar contraste, brillo o cambiar el tono de piel automáticamente.")
        print("3. Presiona la tecla **'r'** para reiniciar todo a la imagen original si quieres empezar de nuevo.")
        print("4. Presiona la tecla **'g'** para GUARDAR la imagen resultante en tu computadora.")
        print("5. Presiona la tecla **'ESC'** para salir del programa.")
        
        while True:
            actualizar_pantalla()
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27: # ESC para salir
                break
            elif key == ord('r'): # Reiniciar
                imagen_trabajo = imagen_original.copy()
                print("[Info] Imagen reiniciada.")
            elif key == ord('g'): # Guardar
                cv2.imwrite("imagen_retocada.jpg", cv2.imread('Estudio de Retoque')) # O guarda la actual
                cv2.imwrite("imagen_retocada.jpg", imagen_trabajo)
                print("[¡Éxito!] Imagen guardada como 'imagen_retocada.jpg' en tu carpeta.")
                
        cv2.destroyAllWindows()