import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import lightgbm as ltb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import pickle
import sys

def prepare_features(data):
    features = []
    overall = []

    for pair in data:
        cos_sim = distance.cosine(pair['vec1'],pair['vec2'])
        man_dist = distance.cityblock(pair['vec1'],pair['vec2'])
        jaccard = distance.jaccard(pair['vec1'],pair['vec2'])
        dice = distance.dice(pair['vec1'],pair['vec2'])

        overall.append(pair['overall'])
        features.append({'cos_sim': cos_sim, 'man_dist': man_dist, 'jaccard': jaccard, 'dice': dice})

    features = pd.DataFrame(features)

    return features, overall


def train_regressor(data, split, model_name, model_save):
    features, overall = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(features, overall, test_size=split)

    if (model_name == 'lgbm'):
        model = ltb.LGBMRegressor() 

    elif (model_name == 'random_forest'):
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pearson = stats.pearsonr(y_test, y_pred)[0]

    importance = model.feature_importances_

    importance = {'cos_sim': importance[0], 'man_dist': importance[1], 'jaccard': importance[2], 'dice': importance[3]}

    joblib.dump(model, model_save, compress=3)

    return model, pearson, importance

def isfloat(num):
    try:
        float(num)
        return True
    
    except ValueError:
        return False


if __name__ == '__main__':

    args = sys.argv

    if (len(args) != 5):
        print('Too many arguments')
        print('Usage: python train_regressor.py (trans|multi) (split) (lgbm|random_forest) (namesave)')
        sys.exit(1)

    if (args[1] != 'trans' and args[1] != 'multi'):
        print('Embedding model does not match to trans | multi')
        print('Usage: python train_regressor.py (trans|multi) (split) (lgbm|random_forest) (namesave)')
        sys.exit(1)

    if (isfloat(args[2]) == False):
        print('Split value must be numeric')
        print('Usage: python train_regressor.py (trans|multi) (split) (lgbm|random_forest) (namesave)')
        sys.exit(1)
    
    if (args[3]!= 'lgbm' and args[3] != 'random_forest'):
        print('Regressor model does not match to lgbm | rf')
        print('Usage: python train_regressor.py (trans|multi) (split) (lgbm|random_forest) (namesave)')
        sys.exit(1)


    bert_model = args[1]
    split = float(args[2])
    regressor = args[3]
    namesave = args[4]

    if (bert_model == 'multi'):
        with open('embedding_train.pkl','rb') as f:
            training_data = pickle.load(f)

    else:
        with open('translation_embedding_train.pkl','rb') as f:
            training_data = pickle.load(f)

    model, pearson, importance = train_regressor(training_data, split, regressor, namesave)

    print("Embedding Model: "+bert_model)
    print("Split: "+str(split))
    print("Regressor: "+regressor)
    print("Pearson Correlation: "+str(pearson))
    print("Importance: ", end="")
    print(importance)