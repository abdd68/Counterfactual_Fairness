import os
import urllib
import os.path
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

dirname = os.path.dirname(__file__)

def read_law_school(label='ZFYA', sensitive_attribute='race', fold=1):
    """
    Load and process the law school dataset.
    
    Args:
        label (str): Name of the target column.
        sensitive_attribute (str): Name of the sensitive attribute column.
        fold (int): Fold for cross-validation (if applicable, currently unused).

    Returns:
        X_train, y_train, sensitive_train, X_test, y_test, sensitive_test
    """
    # Load the dataset
    data = pd.read_csv('data_loading/law_data.csv')
    
    # Shuffle the data
    data = data.sample(frac=1, random_state=fold).reset_index(drop=True)

    # Extract sensitive attribute, target, and features
    y = data[label].values
    sensitive = data[sensitive_attribute].values
    to_drop = [label, sensitive_attribute]

    # Remaining columns are features
    X = data.drop(columns=to_drop).values

    # Split into train and test sets
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=fold
    )

    return X_train, y_train, sensitive_train, X_test, y_test, sensitive_test


def read_dataset(name, label=None, sensitive_attribute=None, fold=None):
    os.chdir('/codespace/fairness/Counterfactual_Fairness')
    if name == 'crimes':
        y_name = label if label is not None else 'ViolentCrimesPerPop'
        z_name = sensitive_attribute if sensitive_attribute is not None else 'racepctblack'
        fold_id = fold if fold is not None else 1
        return read_crimes(label=y_name, sensitive_attribute=z_name, fold=fold_id)
    elif name=='adult':
        return load_adult()
    elif name == 'law_school':
        y_name = label if label is not None else 'ZFYA'
        z_name = sensitive_attribute if sensitive_attribute is not None else 'race'
        fold_id = fold if fold is not None else 1
        return read_law_school(label=y_name, sensitive_attribute=z_name, fold=fold_id)
    else:
        raise NotImplemented('Dataset {} does not exists'.format(name))


def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1):
    os.chdir('/codespace/fairness/Counterfactual_Fairness')
    if not os.path.isfile('communities.data'):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", "communities.data")
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            "communities.names")

    # create names
    names = []
    with open('communities.names', 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv('communities.data', names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(int)

    y = data[label].values
    to_drop += [label]

    z = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    return x[folds != fold], y[folds != fold], z[folds != fold], x[folds == fold], y[folds == fold], z[folds == fold]






#This function is a minor modification from https://github.com/jmikko/fair_ERM
def load_adult(nTrain=None, scaler=True, shuffle=False):
    if shuffle:
        print('Warning: I wont shuffle because adult has fixed test set')
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    if not os.path.isfile('adult.data'):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test")
    data = pd.read_csv(
        "adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        "adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        skiprows=1, header=None
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    #Care there is a final dot in the class only in test set which creates 4 different classes
    target = np.array([-1.0 if (val == 0 or val==1) else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if nTrain is None:
        nTrain = len_train
    data = namedtuple('_', 'data, target')(datamat[:nTrain, :], target[:nTrain])
    data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])

    encoded_data = pd.DataFrame(data.data)
    encoded_data['Target'] = (data.target+1)/2
    to_protect = 1. * (data.data[:,9]!=data.data[:,9][0])

    encoded_data_test = pd.DataFrame(data_test.data)
    encoded_data_test['Target'] = (data_test.target+1)/2
    to_protect_test = 1. * (data_test.data[:,9]!=data_test.data[:,9][0])

    #Variable to protect (9:Sex) is removed from dataset
    return encoded_data.drop(columns=9), to_protect, encoded_data_test.drop(columns=9), to_protect_test
