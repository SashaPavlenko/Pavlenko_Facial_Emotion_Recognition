# Pavlenko_Facial_Emotion_Recognition
 CNN that was created for bachelor's thesis by Aleksandr Pavlenko
Engish version is below

 Для работы с данным репозиторием необходимо установить библиотеки 

* skbuild (pip install scikit-build)
* cv2 (pip install opencv-python)
* torch (pip install torch)


 Для тестирования модели в реальном времени, необходимо в папке Application_for_real_time_testing запустить файл webcam_fer.py

 ## Краткое описание всех папок:
 * Application_for_real_time_testing - папка для тестирование модели в реальном времени. Файл webcam_fer.py принимает на вход видеопоток с веб-камеры и в реальном времени рисует в области всех лиц на изображении прямоугольник с названием эмоции.

 * Best_model - папка содержит файл EmotionClassifier_v16_nc16_ep50.pth, который и является моделью. Более того, в папке приведены логи обучения и визуализация процесса обучения. Из названия модели можно сделать вывод, что модель -- 16 по счету модель класса EmotionClassifier, на вход принимает 16 нейроной, на обучение выделялось 50 эпох.

 * CNN_training - папка с ноутбуком обучения и датасетом.

 * haarcascades - папка, содержащая каскады Хаара, предназначенная для обнаружения лица на видео.

 * Screen_records - папка с записью работы приложения webcam_fer.py

-----

To work with this repository, you need to install libraries

* skbuild (pip install scikit-build)
* cv2 (pip install opencv-python)
* torch (pip install torch)


  To test the model in real time, you need to run the webcam_fer.py file in the Application_for_real_time_testing folder

  ## Brief description of all folders:
  * Application_for_real_time_testing - folder for testing the model in real time. The webcam_fer.py file accepts a video stream from a webcam as input and draws a rectangle with the name of an emotion in the area of all faces in the image in real time.

  * Best_model - the folder contains the EmotionClassifier_v16_nc16_ep50.pth file, which is the model. Moreover, the folder contains training logs and visualization of the training process. From the name of the model, we can get the information that the model is the 16th version model of the EmotionClassifier class, it takes 16 neurons for input, 50 epochs were allocated for training.

  * CNN_training - folder with training notebook and dataset.

  * haarcascades - folder containing haar cascades for face detection in video.
