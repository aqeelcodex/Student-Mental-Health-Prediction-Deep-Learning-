import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

mental = pd.read_csv("deep_learning_seriess/student_mental_health_survey/students_mental_health_survey.csv")
print(mental.head())
print(mental.isnull().sum())
print(mental.dtypes)

mental["CGPA"] = mental["CGPA"].fillna(mental["CGPA"].median())
mental["Substance_Use"] = mental["Substance_Use"].fillna(mental["Substance_Use"].mode()[0])
print(mental.isnull().sum())

categorical_col = mental.select_dtypes(include= "object").columns
for col in categorical_col:
    print(f"{col}: {mental[col].nunique()}")

binary_cols = ["Gender", "Family_History", "Chronic_Illness"]
l_encoders = {}
for col in binary_cols:
    le = LabelEncoder()
    mental[col] = le.fit_transform(mental[col])
    l_encoders[col] = le

ordinal_cols = ["Physical_Activity", "Diet_Quality", "Sleep_Quality",
                "Social_Support", "Substance_Use", 
                "Counseling_Service_Use", "Extracurricular_Involvement"]
ordinal_encoder = OrdinalEncoder()
mental[ordinal_cols] = ordinal_encoder.fit_transform(mental[ordinal_cols])

mental["Relationship_Status"].value_counts(normalize=True)
target = "Relationship_Status"
le_target = LabelEncoder()
y = le_target.fit_transform(mental[target].values)

mental = pd.get_dummies(mental, columns= ["Residence_Type"])
emb_col = "Course"

all_num_cols = ["Age", "Gender", "CGPA", "Stress_Level", "Depression_Score", "Anxiety_Score",
    "Physical_Activity", "Diet_Quality", "Social_Support",
    "Substance_Use", "Counseling_Service_Use", "Family_History", "Chronic_Illness",
    "Financial_Stress", "Extracurricular_Involvement", "Semester_Credit_Load"]

scale = StandardScaler()
mental[all_num_cols] = scale.fit_transform(mental[all_num_cols])
print("scaling is applied")

def create_model(num_numeric_cols, vocab_size, emb_dim, num_classes, emb_col="Course"):
    emb_inputs = []
    emb_layers = []

    inputs = Input(shape=(1,), name=f"{emb_col}_input")
    emb = Embedding(input_dim=vocab_size, output_dim=emb_dim, name=f"{emb_col}_emb")(inputs)
    emb = Flatten()(emb)

    emb_inputs.append(inputs)
    emb_layers.append(emb)

    num_inputs = Input(shape=(num_numeric_cols,), name="numeric_columns")

    x = Concatenate()(emb_layers + [num_inputs])


    x = Dense(256, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)   # 30% dropout
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.25)(x)  # 25% dropout
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)   # 20% dropout
    output = Dense(num_classes, activation="softmax")(x)

    all_inputs = emb_inputs + [num_inputs]
    model = Model(inputs=all_inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

vocab_size = mental["Course"].nunique() + 1
emb_dim = min(50, vocab_size // 2)

X_nums_cols = mental[all_num_cols].values
X_emb_col = mental[emb_col].astype("category").cat.codes.values

s_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracy_scores = []
for train_index, test_index in s_kfold.split(X_nums_cols, y):
    X_train_num, X_test_num = X_nums_cols[train_index], X_nums_cols[test_index]
    X_train_emb, X_test_emb = X_emb_col[train_index], X_emb_col[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = create_model(len(all_num_cols), vocab_size, emb_dim, len(le_target.classes_))
    early_stopping = EarlyStopping(monitor= "val_loss", patience= 5, restore_best_weights= True)
    model.fit([X_train_emb, X_train_num], y_train,     validation_data=([X_test_emb, X_test_num], y_test),
    epochs= 50, batch_size= 32, callbacks= [early_stopping], verbose= 0)

    loss, acc = model.evaluate([X_test_emb, X_test_num], y_test, verbose= 0)
    print(f"Accuracy: {acc:.4f}")
    accuracy_scores.append(acc)
    fold += 1

    print(f"Average accuracy: {np.mean(accuracy_scores)}")
    print("Train Accuracy:", model.evaluate([X_train_emb, X_train_num], y_train, verbose=0)[1])
    print("Test Accuracy:", model.evaluate([X_test_emb, X_test_num], y_test, verbose=0)[1])

