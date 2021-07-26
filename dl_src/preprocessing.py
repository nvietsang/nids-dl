import numpy as np 
import pandas as pd
import tensorflow as tf
import pickle
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def pre_data(df):
    #Add header to Dataframe
    col_names = np.array(['duration', 'protocol_type', 'service', 'flag', 'src_bytes','dst_bytes', 'land', 
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
    'is_host_login','is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
    'srv_rerror_rate', 'same_srv_rate','diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level'])
    new_col = np.array([#'num_conn', 'startTimet', 'orig_pt', 'resp_pt', 'orig_ht', 'resp_ht', 
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes','dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',  
    'is_host_login','is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate','diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate']);
    df.columns = new_col
        # Fix wrong value
    df.su_attempted.replace(2, 0, inplace=True)
    # Delete not useful feature
    # label_column=['num_conn', 'startTimet', 'orig_pt', 'resp_pt', 'orig_ht', 'resp_ht']
    # df = df.drop(label_column, axis=1)
    

    binary_features, categorical_features, numerical_features = get_dtypes()
    numerical_features.remove('num_outbound_cmds')
    data_x_raw =  df.drop('num_outbound_cmds', axis=1)
    
    
    # df['label'] = df.subclass.apply(lambda x: "normal" if x == "normal" else "attack")
    # labels = defaultdict(list)
    # labels['normal'].append('normal')
    # f = open("data/attack_types.txt", "r")
    # for line in f:
    #     subclasses, classes = line.strip().split(' ')
    #     labels[classes].append(subclasses)
    # # mapping attack subclasses to attack classes
    # mapping = dict((v,k) for k in labels for v in labels[k]) 
    # print(dict(labels))

    # attack_label1 = df.subclass.map(lambda x: mapping[x])
    # df['type']=attack_label1



    #  split the test and train into data and labels:
    
    # data_Y = df[label_column]
    # data_x_raw = df.drop(label_column, axis=1)
    # Apply to all features
    for column in data_x_raw.columns:
        if not column in categorical_features:
            minmax_scale_values_df(data_x_raw,column)

    encode_ohe(data_x_raw, categorical_features)

    label_dict = get_label_dict()
    # data_Y['type'] = data_Y.type.apply(lambda x: label_dict[x])
    # data_Y = data_Y['type']

    # print(train_y.unique())
    # for col in data_x_raw.columns:
    #     print(col)
    # print(data_x_raw.columns)

    data_x_raw = data_x_raw.values
    # data_Y = data_Y.values
    return data_x_raw

def get_label_dict():
    label_dict = {'normal': 0, 'dos': 1, 'r2l': 2, 'probe': 3, 'u2r':4}
    return label_dict

def prepro(train, test):
    #Add header to Dataframe
    col_names = np.array(['duration', 'protocol_type', 'service', 'flag', 'src_bytes','dst_bytes', 'land', 
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
    'is_host_login','is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
    'srv_rerror_rate', 'same_srv_rate','diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level'])

    binary_features, categorical_features, numerical_features = get_dtypes()

    train.columns = col_names
    test.columns = col_names
    train['label'] = train.subclass.apply(lambda x: "normal" if x == "normal" else "attack")
    test['label'] = test.subclass.apply(lambda x: "normal" if x == "normal" else "attack")

    print("Train set has " + str(len(train.subclass.unique())) + " subclasses.")
    print("Test set has " + str(len(test.subclass.unique())) + " subclasses.")

    labels = defaultdict(list)
    labels['normal'].append('normal')
    f = open("dl_src/data/attack_types.txt", "r")
    for line in f:
        subclasses, classes = line.strip().split(' ')
        labels[classes].append(subclasses)
    # mapping attack subclasses to attack classes
    mapping = dict((v,k) for k in labels for v in labels[k]) 
    print(dict(labels))

    attack_label1 = train.subclass.map(lambda x: mapping[x])
    train['type']=attack_label1
    attack_label2 = test.subclass.map(lambda x: mapping[x])
    test['type']=attack_label2

    # Fix wrong value
    train.su_attempted.replace(2, 0, inplace=True)
    test.su_attempted.replace(2, 0, inplace=True)
    # Delete not useful feature
    train.drop('num_outbound_cmds', axis=1, inplace=True)
    test.drop('num_outbound_cmds', axis=1, inplace=True)
    numerical_features.remove('num_outbound_cmds')

    #  split the test and train into data and labels:
    label_column=["subclass",'label','difficulty_level','type']
    train_Y = train[label_column]
    train_x_raw = train.drop(label_column, axis=1)

    test_Y = test[label_column]
    test_x_raw = test.drop(label_column, axis=1)

    # Apply to all features
    for column in train_x_raw.columns:
        if column in categorical_features:
            encode_text(train_x_raw,test_x_raw,column)
        else:
            minmax_scale_values(train_x_raw,test_x_raw,column)

    train_x = train_x_raw
    train_y = train_Y

    


    label_dict = {'normal': 0, 'dos': 1, 'r2l': 2, 'probe': 3, 'u2r':4}
    train_y['type'] = train_y.type.apply(lambda x: label_dict[x])
    train_y = train_y['type']
    # print(train_y.unique())

    train_x = train_x.values
    train_y = train_y.values
    print("Information of train data x: ", type(train_x), train_x.shape)
    print("Information of train data y: ", type(train_y), train_y.shape)

    test_x = test_x_raw
    test_y = test_Y
    test_y['type'] = test_y.type.apply(lambda x: label_dict[x])
    test_y = test_y['type']
    # print(test_y.unique())

    test_x = test_x.values
    test_y = test_y.values
    print("Information of test data x: ", type(test_x), test_x.shape)
    print("Information of test data y: ", type(test_y), test_y.shape)

    return train_x, train_y, test_x, test_y

# scaling continous values
def minmax_scale_values_df(training_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    # test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    # testing_df[col_name] = test_values_standardized
def encode_ohe(df, categorical_features):

    with open('dl_src/data/label.txt') as f:
        lines = [line.rstrip() for line in f]
    x = np.array(lines) 
    lines =  np.unique(x)
    # lines = lines.unique()
    for col in lines:
        df[col] = 0
    for col in categorical_features:
        if col == 'service':
            df.drop(col,axis=1, inplace=True)
            continue
        dummy_name = "{}_{}".format(col, df.iloc[0][col])
        df[dummy_name] = 1
        df.drop(col,axis=1, inplace=True)
# one hot encoding
def encode_text_df(training_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)        

        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns :
            testing_df[dummy_name]=testing_set_dummies[x]
        else :
            testing_df[dummy_name]=np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)
    print("Dummy: ", training_df.shape)

# scaling continous values
def minmax_scale_values(training_df,testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized

# one hot encoding
def encode_text(training_df,testing_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        #Print new columns name to file        
        file_label = open("data/label.txt", "a")        
        file_label.write(dummy_name+"\n")
        file_label.close()

        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns :
            testing_df[dummy_name]=testing_set_dummies[x]
        else :
            testing_df[dummy_name]=np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)
    print("Dummy: ", training_df.shape)

def get_dtypes():
    f_names = defaultdict(list)
    t = 0; f = open("dl_src/data/kddcup.names.txt", "r")
    for line in f:
        if (t==0): t = 1;continue
        name, dtype = line.strip()[:-1].split(': ')
        if (name == 'su_attempted' or name == 'root_shell'): continue
        f_names[dtype].append(name)
    f_names['symbolic'].append('root_shell');f_names['symbolic'].append('su_attempted')
    # print(dict(f_names))

    binary_features = ['land',"logged_in","is_host_login",
                       "is_guest_login","root_shell","su_attempted"];
    categorical_features = list(set(f_names['symbolic']) - set(binary_features))
    numerical_features = f_names['continuous']

    return binary_features, categorical_features, numerical_features

if __name__ == '__main__':

    train = pd.read_csv('dl_src/data/KDDTrain+.txt', header=None)
    test = pd.read_csv('dl_src/data/KDDTest+.txt', header=None)
    print("Raw train data has a shape of ", train.shape)
    print("Raw train data has a shape of ", test.shape)

    train_x, train_y, test_x, test_y = prepro(train, test)

    print("Writing train data and test data to file...", end='')
    file_train_x = open("dl_src/data/train_x.pickle", "wb")
    pickle.dump(train_x, file_train_x)

    file_train_y = open("dl_src/data/train_y.pickle", "wb")
    pickle.dump(train_y, file_train_y)

    file_test_x = open("dl_src/data/test_x.pickle", "wb")
    pickle.dump(test_x, file_test_x)

    file_test_y = open("dl_src/data/test_y.pickle", "wb")
    pickle.dump(test_y, file_test_y)

    file_train_x.close()
    file_train_y.close()
    file_test_x.close()
    file_test_y.close()
    print("Done!")
