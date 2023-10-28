### TensorFlow ```saved_model```ni ```tflite``` modelga konvertatsiya qilish va Rasberry PI qurilmasini Hikvision kamerasiga ulagan holda modelni streamlitda deploy qilish dasturi.

#### **Requirements:**
```python
 pip install tensorflow[and-cuda]
```

#### **```saved_model``` ni ```tflite``` modelga konvertatsiya qilish:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model("face_model/1")
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```
