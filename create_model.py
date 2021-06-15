from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle


""" Ради интереса сравним две модели (без особого тюнинга) и лучшую сериализуем для дальнейшего использования """

# Загружаем датасет и делим на тренировочные и тестовые наборы
X, y = load_diabetes(return_X_y=True)

# Упрощаем до одного признака
X = X[:,1].reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=15, random_state=42)

# LinearRegression
lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)

print('LR score: {}'.format(lr_model.score(X_test, y_test))) # 0.41741511122562036
# Дополнительная проверка mae
print('LR mae: {}'.format(mean_absolute_error(y_test, lr_model.predict(X_test)))) # 47.51004513419602

# RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, criterion='mae', max_depth=10, random_state=42)
rf_model = rf.fit(X_train, y_train)

print('RF score: {}'.format(rf_model.score(X_test, y_test))) # 0.5098181046375394
# Дополнительная проверка mae
print('RF mae: {}'.format(mean_absolute_error(y_test, rf_model.predict(X_test)))) # 44.83166666666666

# Лучше результат у RandomForestRegressor, сериализуем
with open('model_regr.pkl', 'wb') as output:
    pickle.dump(rf_model, output)
