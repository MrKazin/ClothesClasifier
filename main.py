from __future__ import absolute_import, division, print_function, unicode_literals
import tkinter as tk                                 # интерфейс
from tkinter import filedialog, Canvas, Frame, BOTH  # получение фотки с винды
from PIL import Image, ImageTk, ImageFilter          # convert and show photos from system
import os                                            # disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'             # disable warnings
import tensorflow as tf
from tensorflow import keras                         # датасеты
import numpy as np                                   # форматирование фоток для tensorflow
import matplotlib.pyplot as plt                      # отрисовка

fashion_mnist = None
train_images = None
train_labels = None
test_images = None
test_labels = None
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
labels_switch = {
        class_names[0]: 0,
        class_names[1]: 1,
        class_names[2]: 2,
        class_names[3]: 3,
        class_names[4]: 4,
        class_names[5]: 5,
        class_names[6]: 6,
        class_names[7]: 7,
        class_names[8]: 8,
        class_names[9]: 9,
    }
model = None
test_loss = None
test_acc = None
my_image = None
my_label = None
my_test_labels = None
variable = None


# ------------------------------------------------------------------------------------------------------------------------------

def trainModel():
    global fashion_mnist
    global train_images
    global train_labels
    global test_images
    global test_labels
    global model
    global test_loss
    global test_acc
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # 28*28 в одномерный
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')  # выходной слой на все типы
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)  # обучение

# ------------------------------------------------------------------------------------------------------------------------------

def uploadPhoto():
    global my_image
    filename = filedialog.askopenfilename(title='Выберите файл.', filetypes=[('Image Files', ['.jpeg', '.jpg', '.png'])])
    img = keras.preprocessing.image.load_img(filename, target_size=(28, 28), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    my_image = img_arr
    image = Image.open(fp=filename)
    image = image.resize((180, 180), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(image)
    panel = tk.Label(root, image=image).grid(row=1, column=3, padx=15)
    root.mainloop()

# ------------------------------------------------------------------------------------------------------------------------------
def setAnswer(string_label):
    global my_label
    global my_test_labels
    try:
        my_label = int(labels_switch[string_label])
    except KeyError as e:
        raise ValueError('Undefined unit: {}'.format(e.args[0]))

    my_test_labels = []
    my_test_labels.insert(0, my_label)
    my_test_labels = np.array(my_test_labels)

# ------------------------------------------------------------------------------------------------------------------------------

def makePredictions():
    global my_test_labels
    predictions = model.predict(my_image)
    print(predictions[0])
    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions, my_test_labels, my_image)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions, my_test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# ------------------------------------------------------------------------------------------------------------------------------

def learnImage():
    global model
    global my_image
    global my_test_labels
    additional_predictions = model.predict(my_image)
    index = my_test_labels[0]
    check_value = additional_predictions[0][index]

    while check_value <= 0.9:
        model.fit(my_image, my_test_labels, epochs=1)
        additional_predictions = model.predict(my_image)
        check_value = additional_predictions[0][index]

# ------------------------------------------------------------------------------------------------------------------------------
def startMakingPredictions():
    global variable
    string_label = variable.get()
    setAnswer(string_label)
    makePredictions()
# ------------------------------------------------------------------------------------------------------------------------------
def startLearningImage():
    global variable
    string_label = variable.get()
    setAnswer(string_label)
    learnImage()
# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Сеть для распознавания типов одежды")
    root.geometry('500x400')

    tk.Label(text="Обучающее множество:").grid(row=0, column=0, sticky=tk.W, pady=10, padx=10)
    buttonDS = tk.Button(text="Обучить", command=trainModel).grid(row=0, column=1)

    tk.Label(text="Загрузка фото:").grid(row=1, column=0, sticky=tk.W, pady=10, padx=10)
    buttonPhoto = tk.Button(text="Загрузить", command=uploadPhoto).grid(row=1, column=1)

    tk.Label(text="Ожидаемый ответ:").grid(row=2, column=0, sticky=tk.W, pady=10, padx=10)

    variable = tk.StringVar(root)
    w = tk.OptionMenu(root, variable, *class_names).grid(row=2, column=1)

    tk.Label(text="Проверка фотографии:").grid(row=3, column=0, sticky=tk.W, pady=10, padx=10)
    buttonPhoto = tk.Button(text="Распознать", command=startMakingPredictions).grid(row=3, column=1)

    tk.Label(text="Запомнить фотографию:").grid(row=4, column=0, sticky=tk.W, pady=10, padx=10)
    buttonAS = tk.Button(text="Запомнить", command=startLearningImage).grid(row=4, column=1)

    root.mainloop()