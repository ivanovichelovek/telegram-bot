import logging

import numpy as np
import requests
from telegram import ext
from PIL import Image
from numpy import asarray
from config import BOT_TOKEN
import os
import tensorflow as tf
from livelossplot import PlotLossesKeras
from inf_of_fr import data
from config import (
    learning_rate,
    verbose,
    image_size,
    input_shape,
    test_dir,
    train_dir,
    output_dir,
)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels = os.listdir(train_dir)
num_classes = len(labels)
last_img = None


"""
учитывая пути к папкам train и test и соотношение проверки к тестированию, этот метод создает три генератора
первый - обучающий генератор использует (100 - validation_percent) изображений из обучающего набора
, он применяет случайные говризонтальные и вертикальные перевороты для увеличения данных 
и генерирует пакеты случайным образом
второй генератор проверки использует оставшийся validation_percent изображений из набора train
, не генерирует случайные пакеты, поскольку модель не обучена на этих данных
точность и потери контролируются с использованием данных проверки, так что скорость обучения может быть обновлена, 
если модель достигает локального оптимума
третий - генератор тестов использует тестовый набор без каких-либо дополнений
. После завершения процесса обучения для этого набора вычисляются окончательные значения точности и потерь.
"""


def build_data_generators(
    train_folder,
    test_folder,
    labels=None,
    image_size=(100, 100),
    batch_size=50,
    val_split=0.001,
):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,  # randomly flip images
        validation_split=val_split,
    )  # augmentation is done only on the train set (and optionally validation)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        train_folder,
        target_size=image_size,
        class_mode="sparse",
        batch_size=batch_size,
        shuffle=True,
        subset="training",
        classes=labels,
    )
    val_gen = train_datagen.flow_from_directory(
        train_folder,
        target_size=image_size,
        class_mode="sparse",
        batch_size=batch_size,
        shuffle=True,
        subset="validation",
        classes=labels,
    )
    test_gen = test_datagen.flow_from_directory(
        test_folder,
        target_size=image_size,
        class_mode="sparse",
        batch_size=batch_size,
        shuffle=False,
        subset=None,
        classes=labels,
    )
    return train_gen, val_gen, test_gen


"""
Создаётся пользовательский слой, который преобразует исходное изображение из
RGB в HSV и оттенки серого и объединяет результаты, 
формирующие входные данные размером 100 x 100 x 4
"""


def convert_to_hsv_and_grayscale(x):
    hsv = tf.image.rgb_to_hsv(x)
    gray = tf.image.rgb_to_grayscale(x)
    rez = tf.concat([hsv, gray], axis=-1)
    return rez


def network(input_shape, num_classes):
    img_input = tf.keras.layers.Input(shape=input_shape, name="data")
    x = tf.keras.layers.Lambda(convert_to_hsv_and_grayscale)(img_input)
    x = tf.keras.layers.Conv2D(
        16, (5, 5), strides=(1, 1), padding="same", name="conv1"
    )(x)
    x = tf.keras.layers.Activation("relu", name="conv1_relu")(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="valid", name="pool1"
    )(x)
    x = tf.keras.layers.Conv2D(
        32, (5, 5), strides=(1, 1), padding="same", name="conv2"
    )(x)
    x = tf.keras.layers.Activation("relu", name="conv2_relu")(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="valid", name="pool2"
    )(x)
    x = tf.keras.layers.Conv2D(
        64, (5, 5), strides=(1, 1), padding="same", name="conv3"
    )(x)
    x = tf.keras.layers.Activation("relu", name="conv3_relu")(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="valid", name="pool3"
    )(x)
    x = tf.keras.layers.Conv2D(
        128, (5, 5), strides=(1, 1), padding="same", name="conv4"
    )(x)
    x = tf.keras.layers.Activation("relu", name="conv4_relu")(x)
    x = tf.keras.layers.MaxPooling2D(
        (2, 2), strides=(2, 2), padding="valid", name="pool4"
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu", name="fcl1")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fcl2")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)
    rez = tf.keras.models.Model(inputs=img_input, outputs=out)
    return rez


"""
этот метод выполняет все шаги, начиная с настройки данных, обучения и тестирования модели и построения графика
результатов модель - это любая обучаемая модель; форма ввода и количество выходных классов зависят от используемого
набора данных, в данном случае входные данные - изображения 100x100 RGB, а выходные данные - слой softmax со 118
вероятностями название используется для сохранения отчета о классификации, содержащего оценку модели f1, графики,
показывающие потери и точность, а также матрицу путаницы размер пакета используется для определения количества
изображений, передаваемых по сети одновременно, количества шагов в epochs выводится из этого как
(общее количество изображений в наборе // размер пакета) + 1
"""


def train_and_evaluate_model(
    model, name="", epochs=25, batch_size=50, verbose=verbose, useCkpt=False
):
    model_out_dir = os.path.join(output_dir, name)
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    if useCkpt:
        model.load_weights(model_out_dir + "/model.h5")

    traingen, valgen, testgen = build_data_generators(
        train_dir,
        test_dir,
        labels=labels,
        image_size=image_size,
        batch_size=batch_size,
        val_split=0.015,
    )

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    save_model = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_out_dir + "/model.h5",
        monitor="loss",
        verbose=verbose,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        traingen,
        epochs=epochs,
        validation_data=valgen,
        steps_per_epoch=(traingen.n // batch_size) + 1,
        verbose=verbose,
        callbacks=[
            early_stopping,
        ],
    )
    return testgen


model = network(input_shape=input_shape, num_classes=num_classes)
testgen = train_and_evaluate_model(model, name="fruit-360 model")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)


async def take_message(update, context):
    global last_img
    # Проверяем, есть ли в сообщении изображения
    if update.message.photo:
        # Получаем список объектов PhotoSize, представляющих изображения разных размеров
        photo_sizes = update.message.photo
        # Выбираем из списка объект с максимальным размером (обычно это последний объект в списке)
        photo = photo_sizes[-1]
        # Скачиваем файл изображения
        file_url = (await context.bot.get_file(photo.file_id)).file_path
        response = requests.get(file_url)
        with open('img.jpg', 'wb') as f:
            f.write(response.content)
        with Image.open("img.jpg") as img:
            img = asarray(img.resize((100, 100))) / 255
            img = np.expand_dims(img, axis=0)
            print(img.shape)
            pred = model.predict(img)
            print(pred.argmax(1))
            last_img = labels[pred.argmax(1)[0]]
            await update.message.reply_text(
                f"Ваш фрукт сорее всего {last_img} из известных нашей программе"
            )
    else:
        await update.message.reply_text("Ваше сообщение не является изображением")


async def help(update, context):
    list_of_comands = (
        "/help - отправляет список комманд с комментариями к их действию\n\n"
        "/ifli - отправляет описание фрукта, изображённого на последней фотографии\n\n"
        "/start - отправляет стартовое сообщение"
    )
    await update.message.reply_text(f"Список комманд: \n{list_of_comands}")


async def ifli(update, context):
    if last_img is None:
        await update.message.reply_text("Вы пока не отправляли изображений")
    else:
        try:
            ret = data[last_img]
            await update.message.reply_text(ret)
        except KeyError:
            await update.message.reply_text("Мы ещё не добавили описание данного фрукта в нашу базу данных")


async def start(update, context):
    """Отправляет сообщение когда получена команда /start"""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет {user.mention_html()}! Я бот по определению экзотических фруктов. "
        rf"Отправьте мне изображение фруктов размером 100х100, и я пришлю название этого фрукта и описание!"
        rf"Для изучения списка комманд напишите /help"
    )


def main():
    application = ext.Application.builder().token(BOT_TOKEN).build()
    application.add_handler(ext.CommandHandler("start", start))
    application.add_handler(ext.CommandHandler("ifli", ifli))
    application.add_handler(ext.CommandHandler("help", help))
    application.add_handler(
        ext.MessageHandler(ext.filters.ALL, take_message)
    )
    application.run_polling()


# Запускаем функцию main() в случае запуска скрипта.
if __name__ == "__main__":
    main()
