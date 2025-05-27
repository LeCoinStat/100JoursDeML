import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def test_logistic_regression_accuracy():
    data_path = Path(__file__).resolve().parents[1] / "05_Apprentissage_Supervise/02_Regression_Logistique/bank_cleaned.csv"
    df = pd.read_csv(data_path, index_col=0)

    X = df.drop(columns=["response", "response_binary"])
    y = df["response_binary"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.7

