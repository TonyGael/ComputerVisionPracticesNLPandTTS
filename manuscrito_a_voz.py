import cv2
import pytesseract
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# archivo con el texto manuscrito
image_path = 'foto_manuscrito.jpg'
output_audio_file = 'texto_leido.mp3'

# cargar y preprocesar la imágen
print(f'Leyenfo la foto: {image_path}')
img = cv2.imread(image_path)

if img is None:
    print(f'Error: No se pudo hallar y cargar la imágen en "{image_path}"'
          'Vuelve a subir la imágen por favor, gracias!')
    exit()

# las imágenes suelen tratarse en escala de grises, mejora el rendimiento hacemos la conversión
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

processed_img = gray_img

# visualizamos lo obtenido hasta el momento
cv2.imshow('Imágen preprocesada:', processed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# realizamos el reconocimiento con tesseract ocr
print('Realizando el reconocimeinto de texto (OCR)...')

extracted_text = pytesseract.image_to_string(processed_img, lang = 'spa')

print('Texto reconocido:')
print(extracted_text)
print("-------------------------")

# limpamos el exto eliminando espacion en blanco inncesarios
# o saltos de linea innecesarios

cleaned_text = os.linesep.join([s for s in extracted_text.splitlines() if s])

if not cleaned_text.strip():
    print('NO se ha detectado texto a generar audio. Subir nuevamente el archivo')
else:
    # convertimos texto a voz
    print('Convirtiendo texto a voz...')
    try:
        tts = gTTS(text=cleaned_text, lang='es', slow=False)
        tts.save(output_audio_file)
        print(f'Audio guardado en: "{output_audio_file}')
        
        # repdoducimos el audio generado
        print("Reproduciendo audio...")
        audio = AudioSegment.from_mp3(output_audio_file)
        play(audio)
        print("Reproducción finalizada.")

    except Exception as e:
        print(f'Error al generar o reproducir e audio: {e}')
