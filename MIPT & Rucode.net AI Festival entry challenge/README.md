# MIPT & Rucode.net AI Festival Entry Challenge

https://rucode.net/iskusstvennyj-intellekt/#intens

It was just 1 out of 15 tasks so I didn't focus on achieving 100% accuracy, but managed to easily score ~95% on test sample by finetuning a pretrained MobileNetV2.

## The task

Компанию RuCode попросили помочь в автоматизации выставления объявлений на одном известном сайте по продаже квартир.

Для этого требуется автоматически определять по фотографии, относится ли она к фотографии здания (0), подъезда (1), интерьера (2) или вида из окна (3).

В дальнейшем на основе этой информации на сайте будет формироваться удобная галерея.
Дана обучающая выборка data.zip с двумя папками с фотографиями: train (обучающая выборка) и test (тестовая выборка). Метки классов для train лежат в файле train.csv.

Вам нужно обучить алгоритм классификации, который для всех изображений из теста будет предсказывать истинный класс.

Качество предсказаний оценивается по метрике Accuracy, умноженной на 10.
Ваш балл будет равен Accuracy, округленной до второго знака после запятой.
https://drive.google.com/drive/folders/1bAQCZXxm-dXiHTrxteMYibOr_echdcKZ?usp=sharing
