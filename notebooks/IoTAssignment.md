This notebook includes:
- Data loading and preprocessing
- Exploratory data analysis
- Feature engineering
- Model training
- Model evaluation

## Collab Code cells
```python
## Data loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/IoT_Intrusion.csv")
df.head()
df.info()

## Label Encoding
tmp_sorted_df = df.groupby(by="label").agg("count").sort_values(by="flow_duration", ascending=False)
labels, counts = tmp_sorted_df["flow_duration"].index, tmp_sorted_df["flow_duration"].values

plt.figure(figsize=(8, 10))
plt.barh(labels, counts)
plt.show()

df.columns

categorical_data = [row for row in df.columns if len(pd.unique(df[row])) <= 2]
categorical_data

df[categorical_data].hist(figsize=(25, 25))

for column in categorical_data:
    print(np.unique(df[column], return_counts=True))

def valid_categorical_features_selector(dataframe, total_categorical_data):
    result = []

    for column in categorical_data:
        elements, counts = np.unique(df[column], return_counts=True)

        if not (elements.shape[0] == 1 or counts[0] < 150 or counts[1] < 150):
            result.append(column)

    return result

valid_categorical_data = valid_categorical_features_selector(df, categorical_data)
valid_categorical_data

df[valid_categorical_data].hist(figsize=(25, 25))

numeric_data = [row for row in df.columns if row not in categorical_data]
numeric_data

df[numeric_data].hist(figsize=(25, 25), bins=3)

# A column will be considered as able to be transformed via standardisation if fewer data instances than the specified threshold are discarded when we trim the data up to 3sigma
def get_columns_for_standard_scaling(dataframe, numeric_cols, threshold):
    result = []

    for column in numeric_cols:
        column_std_3 = 3*dataframe[column].std()
        column_mean = dataframe[column].mean()
        total_data = dataframe[column].count()
        data_after_3_std = dataframe.loc[(dataframe[column]) < (column_mean + column_std_3)][column].count()

        if (total_data - data_after_3_std) < threshold:
            result.append(column)

    return result

valid_numeric_columns_for_standardization = get_columns_for_standard_scaling(df, numeric_data[:-1], 20000)
print(valid_numeric_columns_for_standardization)
print(len(valid_numeric_columns_for_standardization))

def outliers_dropping_based_in_3sigma(dataframe, numeric_cols_to_trim):

    for column in numeric_cols_to_trim:
        column_mean = dataframe[column].mean()
        column_3_sigma = 3*dataframe[column].std()

        dataframe = dataframe[(dataframe[column]) < (column_mean + column_3_sigma)]

    return dataframe

prueba_df = outliers_dropping_based_in_3sigma(df, valid_numeric_columns_for_standardization)

prueba_df[valid_numeric_columns_for_standardization].hist(figsize=(25, 15), bins=25)

all_labels = pd.unique(df["label"])

all_labels

def no_underscore_labels_eraser(dataframe):
  return dataframe[(dataframe["label"].str.find("-") != -1) &
                   (dataframe["label"].str.find("-") != -1) |
                   (dataframe["label"] == "BenignTraffic")]

def label_transformer(label: str):
  if label == "BenignTraffic":
    return label

  character = "-" if "-" in label else "_"

  return label.split(character)[0]

def label_selection(dataframe):
  no_underscore_df = no_underscore_labels_eraser(dataframe)
  label_transformer_vec = np.vectorize(label_transformer)
  return label_transformer_vec(no_underscore_df["label"])

tmp_counter = 0
no_underscore_columns = []

for info in all_labels:

  if info.find("BenignTraffic") != -1:
    continue

  if info.find("-") == -1 and info.find("_") == -1:
    tmp_counter += 1
    no_underscore_columns.append(info)

print(no_underscore_columns)
tmp_counter/all_labels.shape[0]

selected_lables = label_selection(df)
pd.unique(selected_lables)

print(selected_lables.shape, df.shape[0])

df_with_selected_labels = no_underscore_labels_eraser(df)
df_with_selected_labels.loc[:,"final_label"] = selected_lables

df_with_selected_labels.loc[:15, ["label", "final_label"]]

final_label_columns, counts_per_column = np.unique(df_with_selected_labels["final_label"], return_counts=True)

plt.bar(final_label_columns, counts_per_column)
plt.show()

## Train-Test Split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

print(valid_categorical_data)
print(valid_numeric_columns_for_standardization)

len(valid_categorical_data) + len(valid_numeric_columns_for_standardization)

column_trans =  ColumnTransformer(
    [
        ("num", StandardScaler(), valid_numeric_columns_for_standardization),
        ("cat", OrdinalEncoder(), valid_categorical_data)
    ]
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, clone_model, load_model
from keras.regularizers import l2

final_columns = valid_numeric_columns_for_standardization + valid_categorical_data + ["final_label"]
final_columns

X, y = df_with_selected_labels[final_columns[:-1]], df_with_selected_labels[final_columns[-1]]

from sklearn.preprocessing import LabelEncoder

label_enco = LabelEncoder()
y = label_enco.fit_transform(y.values.reshape(-1, 1))
class_names = label_enco.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

num_of_labels = len(np.unique(y_test))

num_of_labels

## Model Training
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, clone_model, load_model
from keras.regularizers import l2

final_columns = valid_numeric_columns_for_standardization + valid_categorical_data + ["final_label"]
final_columns

X, y = df_with_selected_labels[final_columns[:-1]], df_with_selected_labels[final_columns[-1]]

from sklearn.preprocessing import LabelEncoder

label_enco = LabelEncoder()
y = label_enco.fit_transform(y.values.reshape(-1, 1))
class_names = label_enco.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

num_of_labels = len(np.unique(y_test))

num_of_labels

#With dropout and regularization

model_1 = Sequential(
   [
    Input((21,)),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(128, activation="relu", kernel_regularizer=l2(0.1)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu", kernel_regularizer=l2(0.1)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(32, activation="relu", kernel_regularizer=l2(0.1)),
    Dense(16, activation="relu"),
    Dropout(0.1),
    Dense(32, activation="relu", kernel_regularizer=l2(0.1)),
    Dense(64, activation="relu"),
    Dense(num_of_labels, activation="softmax")
    ]
)

#Without dropout or regularization

model_2 = Sequential(
   [
    Input((21,)),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_of_labels, activation="softmax")
    ]
)

#With dropout and without regularization

model_3 = Sequential(
   [
    Input((21,)),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dropout(0.1),
    Dense(32, activation="relu"),
    Dense(64, activation="relu"),
    Dense(num_of_labels, activation="softmax")
    ]
)

def models_trainer(models_container: dict,
                   x_trn,
                   y_trn,
                   epochs: int,
                   checkpoint_path: str = ""
                   ):

  results = {}

  for name, model in models_container.items():

    model = clone_model(model)

    print(f"{name} model in training")

    checkpoint = ModelCheckpoint(
            f"{checkpoint_path}/{name}.keras",
            monitor= 'val_loss',
            verbose= 1,
            save_best_only= True,
        )

    early_stopping = EarlyStopping(
        monitor = "val_loss",
        patience = 10,
        verbose = 2
    )

    model.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics=["accuracy"]
        )

    print(x_trn.shape, y_trn.shape)

    history = model.fit(
                x_trn,
                y_trn,
                batch_size=512,
                epochs=epochs,
                validation_split=0.1,
                callbacks = [checkpoint, early_stopping]
            )

    results[f"{name}"] = (history, model)

  return results

models = {
    "Complete_model": model_1,
    "Most_simple_model": model_2,
    "No_regularization_model": model_3
}

results = models_trainer(models, X_train, y_train, 50, "/content/drive/MyDrive/CS")

for model_name, info_tuple in results.items():
    history = info_tuple[0].history

    print(f"{model_name.upper()}")
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'{model_name} accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'{model_name} accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def generate_confussion_matrix(y_true, y_predict, model_name: str, classes):
    predicted_labels_class = list(map(lambda x: classes[x], y_predict))
    y_true_labels_class = list(map(lambda x: classes[x], y_true))

    df_formatter = pd.DataFrame({
        "Clase predicha": predicted_labels_class,
        "Clase verdadera": y_true_labels_class
    })

    confussion_matrix = pd.crosstab(df_formatter["Clase predicha"],
                                    df_formatter["Clase verdadera"])
    plt.figure(figsize=(17, 5))
    plt.title(model_name)
    sns.heatmap(confussion_matrix, cmap="Blues", annot=True, fmt='g')
    plt.show()

tmp_model = load_model(f"/content/drive/MyDrive/CS/Complete_model.keras")

## Model Evaluation
from sklearn.metrics import classification_report

for model_name in results.keys():
    model = load_model(f"/content/drive/MyDrive/CS/{model_name}.keras")

    print(model_name)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    generate_confussion_matrix(y_test, y_predict, model_name, class_names)
    print(classification_report(class_names[y_test], class_names[y_predict]))
