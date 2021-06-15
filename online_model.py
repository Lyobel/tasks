import numpy as np
import pickle
from flask import Flask, request


# Инициализируем Flask и загружаем модель
app = Flask(__name__)
with open('model_regr.pkl', 'rb') as file:
    model = pickle.load(file)
    

def model_predict(value):
    """ Функция возвращает предсказанное значение """
    # Преобразуем переменную
    value_to_predict = np.array([value]).reshape(-1, 1)
    return model.predict(value_to_predict)[0]

@app.route('/predict')
def predict_func():
    # Считываем значение и преобразуем к float
    value = request.args.get('value')
    try:
        value = float(value)
        return str(model_predict(value))
    except Exception as err:
        # Если произошла ошибка, выводим пользователю
        return 'Error: {}'.format(err)


if __name__ == '__main__':
    app.run('localhost', 5000, debug=False)