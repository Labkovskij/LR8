import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Завантаження даних
data = pd.read_excel('sample_data.xlsx')
description = pd.read_excel('data_description.xlsx')

# Перевірка наявності даних
print(data.head())
print(description.head())

# Перевірка наявності пропущених значень
print("Кількість пропущених значень у кожному стовпці:")
print(data.isnull().sum())

# Вибір ознак і цільової змінної
X = data.drop('loan_closed', axis=1)  # Ознаки
y = data['loan_closed']  # Цільова змінна (бінарна оцінка)

# Вибір лише числових стовпців для нормалізації
X_numeric = X.select_dtypes(include=['float64', 'int64'])

# Обробка пропущених значень за допомогою SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Заповнення середнім значенням
X_numeric_imputed = imputer.fit_transform(X_numeric)

# Перевірка наявності пропущених значень після обробки
print("Кількість пропущених значень у кожному стовпці після обробки:")
print(pd.DataFrame(X_numeric_imputed).isnull().sum())

# Нормалізація ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric_imputed)

# Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Моделювання з використанням Наївного Байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Прогнозування
y_pred = nb_model.predict(X_test)

# Оцінка результатів моделі
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Виявлення шахрайства за допомогою Isolation Forest
# Шахрайство може бути визначено як аномалії у даних
fraud_model = IsolationForest(contamination=0.05, random_state=42)
fraud_model.fit(X_train)

# Прогнозування аномалій
fraud_predictions = fraud_model.predict(X_test)

# Прогноз шахрайства: -1 - аномалія (можливе шахрайство), 1 - нормальний запис
print("Шахрайство (аномалії):")
print(fraud_predictions[:10])

# Переведення результатів у зрозумілий формат
fraud_predictions = ["Шахрайство" if pred == -1 else "Нормально" for pred in fraud_predictions]
print(fraud_predictions[:10])

# Підсумок
print("Завершено виконання скрипту.")
