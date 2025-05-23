import cv2 # Para procesamiento de imágenes
import pytesseract # Interfaz de Python para Tesseract OCR
from gtts import gTTS # Para convertir texto a voz (Google Text-to-Speech)
import os # Para operaciones con el sistema de archivos (guardar/borrar audio)
from pydub import AudioSegment # Para manipular archivos de audio
from pydub.playback import play # Para reproducir el audio con pydub

# --- Configuración General ---
# Asegúrate de que el archivo de imagen 'manuscrito.jpg' esté en la misma carpeta que este script.
ruta_imagen = 'foto_manuscrito.jpg'
nombre_archivo_audio_salida = 'texto_leido.mp3'

# --- 1. Cargar y Preprocesar la Imagen ---
print(f"Cargando imagen desde: {ruta_imagen}")
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print(f"Error: No se pudo cargar la imagen en '{ruta_imagen}'.")
    print("Asegúrate de que la ruta sea correcta y el archivo exista.")
    exit() # Sale del programa si la imagen no se encuentra

# Convertir la imagen a escala de grises para mejorar el reconocimiento OCR.
# La mayoría de los motores OCR funcionan mejor con imágenes en blanco y negro o grises.
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplicar binarización adaptativa para separar el texto del fondo.
# Esto es crucial para texto manuscrito, ya que se adapta a las variaciones de luz.
# - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Utiliza un promedio ponderado gaussiano del área local.
# - cv2.THRESH_BINARY: El tipo de umbral, convierte píxeles por encima del umbral a 255 (blanco), y por debajo a 0 (negro).
# - 11: Tamaño del vecindario (cuadrado de 11x11 píxeles) para calcular el umbral local.
# - 2: Constante sustraída del valor medio ponderado. Ajusta este valor si el texto no se ve bien.
imagen_procesada = cv2.adaptiveThreshold(imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

# --- Opcional: Visualizar la imagen preprocesada (descomentar para depurar) ---
# cv2.imshow('Imagen Preprocesada para OCR', imagen_procesada)
# cv2.waitKey(0) # Espera una tecla para cerrar la ventana
# cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV

# --- 2. Realizar Reconocimiento de Texto (OCR) con Tesseract ---
print("Realizando reconocimiento de texto (OCR)... Esto puede tomar un momento.")
# pytesseract.image_to_string() es la función clave que llama a Tesseract.
# - lang='spa': Especifica que el idioma del texto es español. Asegúrate de tener el paquete 'tesseract-ocr-spa' instalado.
# - config='--psm 6': Page Segmentation Mode (Modo de Segmentación de Página).
#                    --psm 6 asume una sola línea uniforme de texto, lo cual puede ser bueno para manuscritos.
#                    Puedes probar --psm 3 (reconocimiento de página automático) o --psm 4 (columna de texto) si los resultados no son óptimos.
texto_extraido = pytesseract.image_to_string(imagen_procesada, lang='spa', config='--psm 6')

print("\n--- Texto Reconocido por la Máquina ---")
print(texto_extraido)
print("---------------------------------------\n")

# --- 3. Limpiar y Preparar el Texto ---
# Eliminar líneas completamente vacías y espacios en blanco excesivos para mejorar la lectura.
texto_limpio = os.linesep.join([linea for linea in texto_extraido.splitlines() if linea.strip()])

# Verificar si se detectó algún texto significativo
if not texto_limpio.strip(): # .strip() elimina espacios en blanco al inicio/fin
    print("No se detectó texto significativo en la imagen. No se generará audio.")
else:
    # --- 4. Convertir Texto a Voz (TTS) con gTTS ---
    print("Convirtiendo el texto reconocido a voz...")
    try:
        # gTTS necesita conexión a internet para funcionar, ya que usa un servicio web.
        # - text=texto_limpio: El texto que queremos convertir a voz.
        # - lang='es': El idioma de la voz, 'es' para español.
        # - slow=False: Reproducción a velocidad normal.
        voz = gTTS(text=texto_limpio, lang='es', slow=False)
        voz.save(nombre_archivo_audio_salida) # Guarda el audio en un archivo MP3
        print(f"Audio guardado exitosamente en '{nombre_archivo_audio_salida}'")

        # --- 5. Reproducir el Audio ---
        print("Reproduciendo el audio...")
        # Cargar el archivo MP3 con pydub
        audio_segmento = AudioSegment.from_mp3(nombre_archivo_audio_salida)
        # Reproducir el segmento de audio. Requiere 'ffmpeg' y 'simpleaudio' instalados.
        play(audio_segmento)
        print("Reproducción finalizada.")

    except Exception as e:
        print(f"Error al generar o reproducir el audio: {e}")
        print("Asegúrate de tener conexión a internet para gTTS y el paquete 'ffmpeg' instalado para pydub.")

# --- Opcional: Eliminar el archivo de audio después de reproducir ---
# Descomenta la siguiente línea si no quieres guardar el archivo de audio.
# if os.path.exists(nombre_archivo_audio_salida):
#     os.remove(nombre_archivo_audio_salida)
#     print(f"Archivo de audio '{nombre_archivo_audio_salida}' eliminado.")