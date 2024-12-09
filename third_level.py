import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Завантаження даних
data = pd.read_excel('sample_data.xlsx')

# Перевірка наявності пропущених значень
print("Кількість пропущених значень у кожному стовпці:")
print(data.isnull().sum())

# Обробка пропущених значень
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data.select_dtypes(include=['float64', 'int64']))

# Нормалізація даних
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Вибір ознак і цільової змінної
X = data_scaled
y = data['loan_closed']  # Цільова змінна (бінарна оцінка)

# Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Скоринговий аналіз за допомогою випадкового лісу
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Прогнозування
y_pred = rf_model.predict(X_test)

# Оцінка результатів моделі
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Виявлення шахрайства за допомогою Isolation Forest
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