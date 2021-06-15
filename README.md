# Модуль 7. Чем занимается ML-инженер  18. Итоговое задание

## Цель:
- Обучить модель по примеру и сериализовать её.
- Десериализовать модель в коде сервера, загружая её только один раз при старте (это важно, иначе предсказание будет слишком долгим).
- Написать функцию, которая будет принимать запрос с числом, отправлять это число в модель и выводить результат на экран.


## Результат:
- create_model, создание модели и сериализация
- online_model, http сервер на flask.

### Как пользоваться
После запуска приложения online_model.py, в строке браузера ввести требуемое значение, например:
http://localhost:5000/predict?value=-0.04