from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_categorical_columns(df, categorical_cols, ohe_threshold=10):
    
    df_encoded = df.copy()
    label_encoders = {}

    for col in categorical_cols:
        if df_encoded[col].nunique() <= ohe_threshold:
            # One-Hot Encoding (keep all categories)
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
        else:
            # Label Encoding
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    return df_encoded
