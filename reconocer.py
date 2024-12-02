import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import serial
import time

# Mapear los IDs de las clases a etiquetas
labels = ['Mano cerrada (0)', 'Un dedo (1)', 'Dos dedos (2)', 'Tres dedos (3)', 'Cuatro dedos (4)', 'Cinco dedos (5)']

cap = None  # Inicializar 'cap' para evitar advertencias
ser = None  # Inicializar 'ser' para evitar advertencias

try:
    # Cargar el modelo
    model = keras.models.load_model(r'C:\Users\Clases\Desktop\prueba\keras_model.h5')

    # Configurar la conexión serial con Arduino
    ser = serial.Serial('COM3', 9600)  # Asegúrate de que 'COM6' es el puerto correcto
    time.sleep(2)  # Esperar a que la conexión se establezca

    # Intentar abrir la cámara en diferentes índices
    for i in range(3):  # Intenta con los índices 0, 1, 2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Cámara abierta en el índice {i}")
            break
        cap.release()

    if not cap.isOpened():
        print("No se pudo abrir la cámara. Verifica el índice de la cámara y los permisos.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener el cuadro de la cámara.")
            break

        # Preprocesar la imagen para coincidir con Teachable Machine
        img = cv2.resize(frame, (224, 224))  # Ajusta el tamaño según lo que tu modelo espera
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1  # Normalización para coincidir con Teachable Machine
        img = np.expand_dims(img, axis=0)

        # Realizar la predicción
        prediction = model.predict(img)
        class_id = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        # Mostrar la etiqueta y el porcentaje de certeza en la ventana
        label_text = f'{labels[class_id]} ({confidence * 100:.2f}%)'
        cv2.putText(frame, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Video', frame)

        # Enviar el resultado a Arduino si la confianza es >= 80%
        if confidence >= 0.8:
            ser.write(f'{class_id}\n'.encode())
        else:
            ser.write('NoConf\n'.encode())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    ser.close()

except Exception as e:
    print(f'Ocurrió un error: {e}')
    # Liberar recursos en caso de error
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
