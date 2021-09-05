import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def credit_card_risk_prediction(abc):
    credit_df = pd.read_csv('german_credit_data.csv')
    credit_df = credit_df.dropna()
    credit_df.drop(columns=['Unnamed: 0'], inplace=True)

    def outliers(df, fn):
        q1 = df[fn].quantile(0.25)
        q3 = df[fn].quantile(0.75)
        IQR = q3-q1
        lower_bound = q1 - 1.5*IQR
        upper_bound = q3 + 1.5*IQR
        ls = df.index[(df[fn] < lower_bound) | (df[fn] > upper_bound)]
        return ls
    index_list = []
    n = credit_df.select_dtypes(np.number).columns.tolist()
    for i in n:
        index_list.extend(outliers(credit_df, i))

    def remove(df, ls):
        ls = sorted(set(index_list))
        df = df.drop(ls)
        return df
    credit_df = remove(credit_df, index_list)
    from sklearn.model_selection import train_test_split
    train_val_df, test_df = train_test_split(
        credit_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(
        credit_df, test_size=0.25, random_state=42)
    input_col = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
                 'Checking account', 'Credit amount', 'Duration', 'Purpose']
    target_col = 'Risk'
    train_input = train_df[input_col].copy()
    train_target = train_df[target_col].copy()
    val_input = val_df[input_col].copy()
    test_input = test_df[input_col].copy()
    numeric_col = train_input.select_dtypes(np.number).columns.tolist()
    cate_col = train_input.select_dtypes('object').columns.tolist()
    train_input = train_input.dropna(subset=cate_col, how='any')
    val_input = val_input.dropna()
    test_input = test_input.dropna()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler().fit(credit_df[numeric_col])
    train_input[numeric_col] = scaler.transform(train_input[numeric_col])
    val_input[numeric_col] = scaler.transform(val_input[numeric_col])
    test_input[numeric_col] = scaler.transform(test_input[numeric_col])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(
        sparse=False, handle_unknown='ignore').fit(credit_df[cate_col])
    enc_col = encoder.get_feature_names(cate_col).tolist()
    train_input[enc_col] = encoder.transform(train_input[cate_col])
    val_input[enc_col] = encoder.transform(val_input[cate_col])
    test_input[enc_col] = encoder.transform(test_input[cate_col])
    x_train = train_input[numeric_col+enc_col]
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        random_state=42, max_depth=20, max_leaf_nodes=70).fit(x_train, train_target)
    col = numeric_col + enc_col
    a = abc
    inp = dict()
    for i in range(24):
        inp[col[i]] = a[i]
    inp = pd.DataFrame([inp])
    return model.predict(inp)