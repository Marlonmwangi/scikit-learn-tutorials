# IMPORT ENCODER AND TRANSFORMER
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# EXTRACT DATA
Od = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

# REDUCE CABIN CARDINALITY: keep only the deck letter (first char);
# missing cabins become "Missing" so the model still gets a signal.
Od['Deck'] = Od['Cabin'].fillna('Missing').astype(str).str[0]

# FILL THE 2 MISSING 'Embarked' VALUES WITH THE MODE ('S')
Od['Embarked'] = Od['Embarked'].fillna(Od['Embarked'].mode()[0])

# COLUMNS TO ENCODE
cat_cols = ['Deck', 'Sex', 'Embarked']

# ENCODER
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# TRANSFORMER: one-hot the categorical columns, passthrough the rest
transformer = ColumnTransformer(
    transformers=[
        ('encoder', encoder, cat_cols),
    ],
    remainder='passthrough',
    verbose_feature_names_out=False,
)

# FIT + TRANSFORM
transformed = transformer.fit_transform(Od)

# WRAP IN A LABELED DATAFRAME
out = pd.DataFrame(
    transformed,
    columns=transformer.get_feature_names_out(),
)

print('Shape :', out.shape)
print('Encoded columns :', [c for c in out.columns if any(c.startswith(p) for p in cat_cols)])
print('Passthrough cols:', [c for c in out.columns if not any(c.startswith(p) for p in cat_cols)])
print()
print('First row (transposed, so each column is visible):')
print(out.iloc[0].T)
