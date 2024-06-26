# PyAra Library Documentation

```
pyara/
├── Model/
│   ├── Model_weights.bin
│   └── __init__.py
├── audio_prepare.py
├── config.py
├── main.py
├── model.py
├── tests/
│   └── test_audio/
│       ├── 1.aiff
│       ├── 1.flac
│       ├── 1.mp3
│       ├── 1.ogg
│       ├── 1.wav
│       ├── 2.wav
│       ├── Alg_1_0.wav
│       └── real_0.wav
│   └── tests.py
├── Developer_Guide.odt
├── Developer_Guide.pdf
├── setup.cfg
├── setup.py
└── LICENSE
```



## Namespaces

`pyara` - основной пространство имен для работы с аудиоанализом.

## Classes

### CFG

Класс конфигурации с основными переменными для управления поведением модели.

**Public Members:**
- `JUST_PREDICT`: bool - Включить только предсказание, без тренировки.
- `DEBUG`: bool - Режим отладки.
- `DATASET`: string - Имя набора данных.
- `Docker`: bool - Флаг использования Docker.
- `visualize`: bool - Включить визуализацию.
- `FULL_DATA`: bool - Использовать полный набор данных.
- `wandb_on`: bool - Включение Weight & Biases для отслеживания.
- `seed`: int - Сид для генерации случайных чисел.
- `model_name`: string - Имя модели.
- `info`: string - Информация о конфигурации.
- `archive_name`: string - Имя архива данных.
- `width`: int - Ширина сигнала.
- `train_bs`: int - Размер батча при обучении.
- `valid_bs`: int - Размер батча при валидации.
- `mels`: int - Количество Mel-кепстральных коэффициентов.
- `lstm_layers`: int - Количество слоев LSTM.
- `SAMPLE_RATE`: int - Частота дискретизации.
- `NUM_SAMPLES`: int - Количество выборок.
- `num_item_all`: int - Общее количество элементов.
- `num_test`: int - Количество тестовых элементов.
- `num_classes`: int - Количество классов.
- `classes`: list - Список классов.
- `activation`: string - Функция активации.
- `device`: torch.device - Выбранное устройство для вычислений.
- `best_model_w`: string - Имя файла с лучшими весами модели.

## Functions

### predict_audio

Предсказывает тип аудио (подлинное или синтезированное).

**Parameters:**
- `file_path`: string - Путь к аудиофайлу.
- `print_probability`: bool - Флаг вывода вероятности.
- `pitch_shift`: int - Смещение тональности.
- `sample_rate`: int - Частота дискретизации.
- `width`: int - Ширина окна.

**Returns:**
- `int`: 0 для подлинного аудио, 1 для синтезированного.

**Example:**

from pyara import predict_audio

predict_audio('C:/Tests/mp3.mp3')


### Функция cut_if_necessary

### Описание
Функция `cut_if_necessary` предназначена для обрезки аудиосигнала до заданной ширины (количества выборок), если исходный сигнал имеет большую длину.

### Входные данные
- `signal` (`torch.Tensor`): Аудиосигнал.
- `width` (`int`): Ширина сигнала после обрезки. По умолчанию `CFG.width = 400`.

### Выходные данные
- `torch.Tensor`: Обрезанный аудиосигнал.

### Описание работы функции
- Проверяет длину аудиосигнала (по оси времени, представленной третьим измерением тензора).
- Если длина аудиосигнала превышает заданную ширину (`width`), то происходит обрезка, удаляя лишние выборки.
- Возвращает обрезанный аудиосигнал.

### Пример использования

```python
import torchaudio
import torch
MFCC_spectrogram = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=80,
    melkwargs={
        "n_fft": 1024,
        "n_mels": 80,
        "hop_length": 256,
        "mel_scale": "htk",
        'win_length': 1024,
        'window_fn': torch.hann_window,
        'center': False
    },
)
from pyara import cut_if_necessary
signal = signal.mean(dim=0)
signal = signal.unsqueeze(dim=0)
signal, sample_rate = torchaudio.load('test_audio')
signal = MFCC_spectrogram(signal)
signal = cut_if_necessary(signal, 300)
```

## Функция `right_pad_if_necessary`

Функция `right_pad_if_necessary` выполняет дополнение последнего измерения входного сигнала вправо, если это необходимо.

### Входные данные:
- `signal` (`torch.Tensor`): Тензор сигнала, представляющий многомерные данные. Должен быть трехмерным тензором с размерностью `[batch_size, num_channels, signal_length]`.
- `width` (`int`): Желаемая ширина сигнала после дополнения. По умолчанию значение берется из `CFG`.

### Выходные данные:
- `torch.Tensor`: Дополненный сигнал с выполненным дополнением последнего измерения вправо.

### Пример использования:

```python
import torchaudio
import torch
MFCC_spectrogram = torchaudio.transforms.MFCC(
    sample_rate=16000,
    n_mfcc=80,
    melkwargs={
        "n_fft": 1024,
        "n_mels": 80,
        "hop_length": 256,
        "mel_scale": "htk",
        'win_length': 1024,
        'window_fn': torch.hann_window,
        'center': False
    },
)
from pyara import right_pad_if_necessary
signal, sample_rate = torchaudio.load('C:/Users/79671/Downloads/real_0.wav')
signal = signal.mean(dim=0)
signal = signal.unsqueeze(dim=0)
signal = MFCC_spectrogram(signal)
signal = right_pad_if_necessary(signal, 300)
signal.shape
```

## 2.4 Функция `prepare_signal`

Функция `prepare_signal` используется для подготовки аудиосигнала к обработке нейронной сетью.

### Входные данные:
- `voice_path`: Путь к аудиофайлу;
- `pitch_shift`: Смещение тональности аудио (по умолчанию 0);
- `width`: Ширина сигнала после обрезки;
- `sample_rate`: Частота дискретизации аудиосигнала.

### Выходные данные:
- `signal`: Подготовленный сигнал для обработки.

### Описание работы функции:
1. Загрузка аудиофайла;
2. Усреднение сигнала по каналам;
3. Добавление размерности пакета к сигналу;
4. Применение сдвига тональности, если требуется;
5. Применение преобразования MFCC;
6. Обрезка и дополнение спектрограммы, при необходимости;
7. Повторение спектрограммы;
8. Добавление размерности пакета к спектрограмме;
9. Перемещение спектрограммы на устройство из CFG;
10. Возврат подготовленного сигнала.

### Пример использования:
```python
from pyara import prepare_signal 
prepare_signal('C:/Tests/mp3.mp3')
```

## 2.5 Функция `prepare_signals`

Функция `prepare_signals` используется для подготовки нескольких аудиосигналов к обработке нейронной сетью.

### Входные данные
- `voice_paths`: Список путей к аудиофайлам;
- `pitch_shift`: Смещение тональности аудио (по умолчанию 0);
- `width`: Ширина сигнала после обрезки;
- `sample_rate`: Частота дискретизации аудиосигнала.

### Выходные данные
- `signals`: Список подготовленных сигналов для обработки.

### Описание работы функции
Для каждого аудиофайла в `voice_paths`:
1. Загрузка и усреднение сигнала;
2. Добавление размерности пакета;
3. Применение сдвига тональности, если необходимо;
4. Применение преобразования MFCC;
5. Обрезка и дополнение спектрограммы;
6. Повторение спектрограммы;
7. Добавление размерности пакета;
8. Перемещение на устройство из CFG;
9. Сборка всех подготовленных сигналов в `res`.

### Пример использования

```python
from pyara import prepare_signals 
prepare_signals(['C:/Tests/mp3.mp3', 'C:/Tests/mp23.mp3'])
```

## 2.6 Функция `prediction`

Функция `prediction` используется для получения предсказания от модели по входному сигналу.

### Входные данные:
- `model`: Модель для предсказания, совместимая с PyTorch;
- `signal`: Входной сигнал, трехмерный тензор `[batch_size, num_channels, signal_length]`;
- `print_probability`: Флаг для вывода вероятности предсказания (по умолчанию `False`).

### Выходные данные:
Возвращает кортеж с предсказанной меткой класса и вероятностью. При `print_probability=True` выводит также строку с вероятностью.

### Описание работы функции:
1. Перенос модели на устройство из CFG;
2. Сжатие измерения сигнала;
3. Прямой проход модели;
4. Применение Softmax для вероятностей;
5. Возврат метки класса и вероятности;
6. Вывод вероятности, если требуется.

### Пример использования:

```python
from pyara import prediction
model = model_eval()
signal = prepare_signal(file_path, pitch_shift, width, sample_rate)
prediction(model, signal)
```
## 2.7 Функция `prediction_multiple`

Функция `prediction_multiple` используется для получения предсказаний от модели по нескольким входным сигналам.

### Входные данные:
- `model`: Модель для предсказания, совместимая с PyTorch;
- `signals`: Список входных сигналов, каждый из которых - трехмерный тензор `[batch_size, num_channels, signal_length]`.

### Выходные данные:
Возвращает список предсказанных меток классов (1 для синтезированного голоса, 0 для реального голоса) и список вероятностей предсказания для каждого сигнала.

### Описание работы функции:
1. Перенос модели на устройство из CFG;
2. Проход по всем сигналам для сжатия измерений, прямого прохода модели и применения Softmax;
3. Определение предсказанных классов и сбор вероятностей;
4. Возврат списков классов и вероятностей.

### Пример использования:

```python
from pyara import prediction_multiple
model = model_eval()
signals = prepare_signals([file_path1, file_path2], pitch_shift, width, sample_rate)
prediction_multiple(model, signals)
```

## Функция: `model_eval`

Эта функция выполняет следующие задачи:

1. **Создание экземпляра `MFCCModel`**:
   - Инициализирует новый экземпляр класса `MFCCModel`.

2. **Загрузка весов модели**:
   - Определяет путь к файлу с весами модели, обычно называемому 'Model_weights.bin', который находится в той же директории, что и скрипт.
   - Загружает веса модели в созданный экземпляр `MFCCModel`.

3. **Перевод модели в режим оценки**:
   - Переводит модель в режим оценки с помощью метода `eval()`. Это необходимо для тестирования или использования модели, поскольку отключает определенные слои и поведения, такие как слои исключения и нормализация пакетов, активные только во время обучения.

4. **Перемещение модели на указанное устройство**:
   - Перемещает модель на устройство (например, CPU, GPU), указанное в конфигурации `CFG.device`.

5. **Возврат модели**:
   - Выводит сообщение 'Model Evaluated!' для указания на успешное завершение процесса.
   - Возвращает готовый к использованию и перенесенный на указанное устройство экземпляр `MFCCModel`.

### Возвращаемое значение
- `MFCCModel`: Экземпляр класса `MFCCModel`, который был загружен весами, переведен в режим оценки и перемещен на указанное устройство.

