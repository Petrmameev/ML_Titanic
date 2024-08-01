import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings("ignore")

# Загрузка данных
train_data = pd.read_csv('train(titanic).csv')
test_data = pd.read_csv('test(titanic).csv')
submission_data = pd.read_csv('gender_submission(titanic).csv')

# Фильтрация данных
filtered_train_data = train_data[~train_data['Embarked'].isin(['C', 'S', 'Q'])]

# Заполнение пропусков в 'Embarked'
embarked_mode = train_data['Embarked'].mode()[0]
train_data['Embarked'].fillna(embarked_mode, inplace=True)

# Функция для извлечения титулов и кодирования
def extract_and_encode_titles(data):
    data['Title'] = data.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
    data.Title = data.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                      'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data.Title = data.Title.replace('Mlle', 'Miss')
    data.Title = data.Title.replace('Ms', 'Miss')
    data.Title = data.Title.replace('Mme', 'Mrs')

    encoder_title = OneHotEncoder(sparse_output=False, drop='first')
    title_encoded = encoder_title.fit_transform(data[['Title']])
    title_encoded_df = pd.DataFrame(title_encoded, columns=encoder_title.get_feature_names_out(['Title'])).astype(int)
    data = pd.concat([data, title_encoded_df], axis=1)
    data.drop('Title', axis=1, inplace=True)

    encoder_embarked = OneHotEncoder(sparse_output=False, drop='first')
    embarked_encoded = encoder_embarked.fit_transform(data[['Embarked']])
    embarked_encoded_df = pd.DataFrame(embarked_encoded, columns=encoder_embarked.get_feature_names_out(['Embarked'])).astype(int)
    data = pd.concat([data, embarked_encoded_df], axis=1)

    encoder_sex = OneHotEncoder(sparse_output=False, drop='first')
    sex_encoded = encoder_sex.fit_transform(data[['Sex']])
    sex_encoded_df = pd.DataFrame(sex_encoded, columns=encoder_sex.get_feature_names_out(['Sex'])).astype(int)
    data = pd.concat([data, sex_encoded_df], axis=1)

    encoder_pclass = OneHotEncoder(sparse_output=False, drop='first')
    pclass_encoded = encoder_pclass.fit_transform(data[['Pclass']])
    pclass_encoded_df = pd.DataFrame(pclass_encoded, columns=encoder_pclass.get_feature_names_out(['Pclass'])).astype(int)
    data = pd.concat([data, pclass_encoded_df], axis=1)

    data.drop(['Sex', 'Pclass', 'Embarked', 'Name', 'PassengerId', 'Cabin', 'Ticket', 'Age'], axis=1, inplace=True)

    return data

# Применение функции к данным
train_data = extract_and_encode_titles(train_data)
test_data = extract_and_encode_titles(test_data)

# Заполнение пропусков в 'Fare' для тестовых данных
test_data['Fare'] = test_data['Fare'].fillna(np.mean(test_data['Fare']))

# Добавление новых признаков
def add_family_features(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0
    data['FarePerPerson'] = data['Fare'] / data['FamilySize']
    data.drop('Fare', axis=1, inplace=True)

    return data

# Применение функции для добавления признаков
train_data = add_family_features(train_data)
test_data = add_family_features(test_data)

# Удаление ненужных столбцов
train_data.drop(['Title_Rare', 'FamilySize', 'Title_Mr', 'Pclass_2', 'Title_Miss'], axis=1, inplace=True)
test_data.drop(['Title_Rare', 'FamilySize', 'Title_Mr', 'Pclass_2', 'Title_Miss'], axis=1, inplace=True)

# Подготовка данных для обучения
X_train = train_data.drop('Survived', axis=1)  # Признаки
y_train = train_data['Survived']  # Целевая переменная

# Обучение модели
logistic_regression_model = LogisticRegression(solver='liblinear', max_iter=100)
logistic_regression_model.fit(X_train, y_train)

# Функция для k-fold кросс-валидации
def perform_k_fold_cross_validation(model, X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return scores

# Применение кросс-валидации
cv_scores = perform_k_fold_cross_validation(logistic_regression_model, X_train, y_train, num_folds=5)
print(f'Оценки кросс-валидации: {cv_scores}')
print(f'Средняя точность кросс-валидации: {np.mean(cv_scores):.4f}')


# Прогнозирование на тестовых данных
predictions = logistic_regression_model.predict(test_data)
print(test_data.columns)

# Оценка модели с использованием выходных данных (истинные значения)
true_survival_values = submission_data['Survived']
accuracy = accuracy_score(true_survival_values, predictions)
roc_auc = roc_auc_score(true_survival_values, predictions)

print(f"Точность модели на тестовых данных: {accuracy * 100:.4f}%")
print(f"ROC AUC модель на тестовых данных: {roc_auc:.4f}")
print("Матрица ошибок:")
print(confusion_matrix(true_survival_values, predictions))