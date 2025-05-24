import cv2
# import pytesseract
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# importamos keras_ocr
import keras_ocr

# archivo con el texto manuscrito
ruta_imagen = 'foto_manuscrito.jpg'
nombre_audio_salida = 'texto_leido_keras_ocr.mp3'

# cargar y preprocesar la imágen
print(f'Leyendo la foto: {ruta_imagen}')
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f'Error: No se pudo hallar y cargar la imágen en "{ruta_imagen}"'
          'Vuelve a subir la imágen por favor, gracias!')
    exit()

# keras usa rgb y open bgr
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# PIPELINE KERAS Y DESCARGA DE MODELOS PREENTRENADO INCIAL
print('Inicializando pipeline de keras-ocr (puede descargar modelos la primera vez)...')
pipeline = keras_ocr.pipeline.Pipeline()

# detecion y reconocimiento con keras-ocr
print('Realizando detección y reconocimiento de texto con keras-ocr...')
predicciones = pipeline.recognize([imagen_rgb])

# extraccion
texto_extraido_palabras = [palabra for palabra, _ in predicciones[0]]
texto_extraido = ' '.join(texto_extraido_palabras)

print('\n--- Texto Reconocido por la Máquina (Keras OCR) ---')
print(texto_extraido)
print('---------------------------------------------------\n')

# limpiamos el texto
texto_limpio = texto_extraido.strip()

if not texto_limpio:
    print("No se detectó texto significativo. No se generará audio.")
else:
    # Texto a Voz (TTS) con gTTS
    print("Convirtiendo el texto reconocido a voz...")
    try:
        voz = gTTS(text=texto_limpio, lang='es', slow=False)
        voz.save(nombre_audio_salida)
        print(f'Audio guardado exitosamente en "{nombre_audio_salida}"')

        # reproducir el Audio
        print('Reproduciendo el audio...')
        audio_segmento = AudioSegment.from_file(nombre_audio_salida)
        play(audio_segmento)
        print('Reproducción finalizada.')

    except Exception as e:
        print(f'Error al generar o reproducir el audio: {e}')
        print('Asegúrate de tener conexión a internet para gTTS y el paquete "ffmpeg" instalado para pydub.')