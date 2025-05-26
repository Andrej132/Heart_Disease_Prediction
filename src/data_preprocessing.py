import pandas as pd
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from config import MODEL_DIR, DATA_DIR


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def preprocess_data(df):
    numerical = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR']
    normalize_col = ['Oldpeak']
    categorical = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina']
    ordinal_col = ['ST_Slope']
    st_slope_order = [['Down', 'Flat', 'Up']]

    df = df.copy()

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('norm', MinMaxScaler(), normalize_col),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical),
        ('ord', OrdinalEncoder(categories=st_slope_order), ordinal_col)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(preprocessor, f"{MODEL_DIR}/preprocessor.pkl")

    feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    train_data = pd.DataFrame(X_train, columns=feature_names)
    train_data['HeartDisease'] = y_train.values

    test_data = pd.DataFrame(X_test, columns=feature_names)
    test_data['HeartDisease'] = y_test.values

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    train_data.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test_data.to_csv(f"{DATA_DIR}/test.csv", index=False)

