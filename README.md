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
Kameraning kerakli konfiguratsiyalarini olish uchun [SADP](https://www.hikvision.com/en/support/tools/hitools/clea8b3e4ea7da90a9/) dasturini yuklab oling.

Kamerada dastur yordamida foydalanish uchun user name, password, ip address va rtsp port kerak bo'ladi.
```python
camera = f"rtsp://{user}:{password}@{ip_address}:{rtsp}/h264/ch1/main/av_stream"
```
user va password SADP dagi login va parol hisoblanadi.

```ip_address```ni quyidagi bo'limdan olishingiz mumkin:
![sadp2](https://github.com/MisterFoziljon/Deploy-TF-Lite-model-with-RasberryPI-device-using-hikvision-camera/blob/main/images/sadp2.png)
