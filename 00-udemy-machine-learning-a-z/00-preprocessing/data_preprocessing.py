import pandas as pd

dataset = pd.read_csv('Data.csv')
# matrix of features
X = dataset.iloc[:, :-1].values
# dependent variable vector
Y = dataset.iloc[:, 3].values

# missing data - take the mean of the columns

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
onehoteencoder = OneHotEncoder(categorical_features = [0])
X = onehoteencoder.fit_transform(X).toarray()

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# Splitting the dataset into traning and testing
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
