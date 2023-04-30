import os

BOT_TOKEN = "5613665398:AAHwMdLkrWfDZyjCGWMVS2wIi2Boub9d7CE"

learning_rate = 0.1
# изначальный learning rate
min_learning_rate = 0.00001
# как только learning rate достигнет этого значения, не снижайть ее дальше
learning_rate_reduction_factor = 0.5
# коэффициент, используемый при снижении learning rate -> learning_rate *= learning_rate_reduction_factor
verbose = 2
# контролирует объем протоколирования, выполняемого во время обучения и тестирования: 0 - нет,
# 1 - сообщает показатели после каждого пакета, 2 - сообщает показатели после каждой эпохи
image_size = (100, 100)
# ширина и высота используемых изображений
input_shape = (
    100,
    100,
    3,
)
# ожидаемая форма ввода для обученных моделей (поскольку изображения в Fruit-360
# представляют собой изображения 100 x 100 RGB, это требуемая форма ввода)
base_dir = "Fruit-Images-Dataset"  # относительный путь к папке Fruit-Images-Dataset
test_dir = os.path.join(base_dir, "Test")
train_dir = os.path.join(base_dir, "Training")
output_dir = "output_files"
# корневая папка, в которую будут сохранены выходные файлы; файлы будут находиться в разделе output_files/model_name