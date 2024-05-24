import pandas as pd
from sklearn.model_selection import train_test_split

def get_participants(participants_path):

    # all participants dataframe
    df = pd.read_csv(participants_path, sep='\t')

    #put labels - fcd group gets label 0, hc group gets label 1
    list_labels = []
    for i in range(len(df)):
        if df.iloc[i]['group'] == 'fcd':
            list_labels.append(0)
        else:
            list_labels.append(1)
    df_labels = pd.DataFrame(list_labels, columns = ['label'])
    df = pd.concat([df, df_labels], axis = 1)

    # train/test participants
    dataset_train, dataset_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    dataset_train = dataset_train.sample(frac=1).reset_index(drop = True)
    dataset_test = dataset_test.sample(frac=1).reset_index(drop = True)

    # print("GetParticipants is finished" , flush=True)
    return dataset_train, dataset_test

