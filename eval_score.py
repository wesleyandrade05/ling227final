from train_regressor import prepare_features
from train_bert import bert_embedder
from scipy import stats
import numpy as np
import pickle
import joblib
import sys


def score(model, eval_data):
    eval_features, eval_overall = prepare_features(eval_data)

    eval_pred = model.predict(eval_features)

    pearson = stats.pearsonr(eval_overall, eval_pred)[0]

    return pearson


if __name__ == '__main__':

    args = sys.argv

    pairs = ['ar-ar','de-de','de-en','de-fr','de-pl','en-en','es-es','es-en','es-it','fr-fr','fr-pl','it-it','pl-pl','pl-en','ru-ru','tr-tr','zh-zh','zh-en']

    if (len(args) > 5):
        print("Too many arguments")
        print("Usage: python eval_score.py (trans|multi) modelfile lang1 lang2")
        sys.exit(1)

    if (args[1] != "multi" and args[1] != "trans"):
        print("Embedding Model does not match")
        print("Usage: python eval_score.py (trans|multi) modelfile lang1 lang 2")
        sys.exit(1)

    bert_model = args[1]
    modelfile = args[2]

    if (bert_model == 'multi'):
        with open('embedding_eval.pkl','rb') as f:
            eval_data = pickle.load(f)

    else:
         with open('translation_embedding_eval.pkl','rb') as f:
            eval_data = pickle.load(f)       

    if (args[3] != 'all'):
        if (args[3]+'-'+args[4] not in pairs):
            print('Pair of languages not found')
            sys.exit(1)
        else:
            lang1 = args[3]
            lang2 = args[4]

            final_evaluation_data = []

            for pair in eval_data:
                if (pair['lang1'] == lang1 and pair['lang2'] == lang2):
                    final_evaluation_data.append(pair)

            eval_data = final_evaluation_data

    model = joblib.load(modelfile)

    score = score(model, eval_data)

    print("Pearson Score: "+str(score))
