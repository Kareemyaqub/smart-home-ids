# Methodology

## Data Preprocessing
The dataset is cleaned by handling missing values, encoding labels, and scaling numerical features.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Feature Engineering
Correlation analysis and feature importance techniques are applied to reduce redundancy.
```
importances = model.feature_importances_
```

## Model Training
Multiple supervised learning algorithms were implemented and compared.
```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Evaluation
Performance is assessed using multiple metrics.
```
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
```
