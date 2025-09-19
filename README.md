Student Mental Health Prediction (Deep Learning)

This project applies a deep learning model to predict student mental health–related outcomes using survey data. The model combines numeric features (e.g., Age, CGPA, Stress Level) and an embedding layer for the categorical feature Course.

We experimented with regularization (Dropout + L2) and Stratified K-Fold Cross Validation to ensure fair model evaluation.

🔑 Key Steps

Data Preprocessing

Missing values imputed with median (for numeric) and mode (for categorical).

Binary columns encoded with LabelEncoder.

Ordinal columns encoded with OrdinalEncoder.

Other categorical columns converted with pd.get_dummies.

Numeric features standardized using StandardScaler.

Model Architecture

Embedding layer for the Course column.

Dense layers with ReLU activation, L2 regularization, and Dropout to reduce overfitting.

Output layer with softmax for multi-class classification.

Evaluation

5-Fold Stratified Cross Validation.

Compared train vs. test accuracy to detect overfitting/underfitting.

📊 Results

Train and test accuracies remain very close (~0.50) across folds.

This shows the model is neither overfitting nor underfitting.

The low accuracy is primarily due to noisy or weakly correlated data, not the model itself.
