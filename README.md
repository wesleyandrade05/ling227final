# LING 227 FINAL PROJECT - Multilingual News Articles Similarity

This Github contains the code and documentation from the Final Project developed by Wesley Andrade and Lasse van den Berg on Yale's Class LING 227 of Spring 2023. Our goal was to reproduce, according to the context of the class, the [Task 8](https://competitions.codalab.org/competitions/33835#learn_the_details-task-results) proposed on the SemEval 2022. A thorough description of the task can be found on the provided link. The general approaches that the participants took during the competition and the results obtained can be found on this [paper](https://aclanthology.org/2022.semeval-1.155.pdf).

Our approach was to use a pre-trained BERT model through a bi-encoder architecture, where each newspaper article would go through a BERT encoder and have its embedding generated. Then, we combined the two embeddings into features and used two Decision Trees Regressors ([LGBM](https://lightgbm.readthedocs.io/en/v3.3.2/) and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)) to obtain the respective score.

The results we obtained and a more complete description of our approach are in the PDF attached to this repository.

<br>

## How to use the code

<br>

### Prepare Data

The code has 4 main python files, which are the ones we used to prepare the data, perform the training and generate the evaluation scores. First, we used a [script](https://github.com/euagendas/semeval_8_2022_ia_downloader) provided by the organization that retrieved the information from the url links provided and uploaded json files containing them to a folder.

The first command line to be executed is 
```console
python train_bert.py model
```

Here, you specify to which model you want to generate the embeddings and if it is for the training or evaluation dataset. The accepted arguments are *t_multi*, *t_trans*, *e_multi*, *e_trans*, where *e* stands for evaluation, *t* for training, and *trans* for the pre-trained [bert-large-cased](https://huggingface.co/bert-large-cased) model and *multi* for the pre-trained [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased) model. However, since running it takes a lot of time (especially for the *trans* model where we have to translate each non-english text using [googletrans](https://pypi.org/project/googletrans/)), we already provided the files containing the embeddings and the required data for the regressor to run. These files are on .pkl format and are called *embedding_eval.pkl*, *embedding_train.pkl*, *translation_embedding_eval.pkl*, *translation_embedding_train.pkl*.

The articles and the files can be found on a [Google Drive folder](https://drive.google.com/drive/folders/1We5Up6zBFChUwJv19QFm1eeCoZfvdtT8?usp=sharing), since they were very large to be uploaded to Github.

<br>

### Train the Regressor

To train the regressor, you should use the following command:

```console
python train_regressor.py (trans|multi) (split) (lgbm|random_forest) (namesave)
```

The (trans | multi) argument specifies which model you will use to train, the (split) should be a float value between 0 and 1 that specifies the size of the validation dataset, (lgbm | random_forest) specifies which Regressor Model you want to use, and (namesave) should be the name of the file to which the trained regressor will be restored. Preferentially, you should save it as a .joblib file, since this is the package which we are using to store the regressors. Besides that, the model assumes that the datsets are saved with the names given on the previous section. So, they should not be changed.

Running this command will return the Pearson Correlation of the regressor on the validation dataset and the overall importance of each of the features. The ones for the LGBM are not normalized.

<br>

### Evaluation Dataset Score

Now, having the trained model and the mbedding for the evaluation dataset, we can run:
```console
python eval_score.py (trans|multi) modelfile lang1 lang2
```

The (trans | multi) argument again determines which embedding model you will be using, and the (moddelfile) loads the regressor file. The arguments (lang1) and (lang2) specify on which pair of languages we want to test our model. They can be either *all all* if we want to test on the entire evaluation dataset, or we can specify by using one of the pair of languages provided:
```python
pairs = ['ar-ar','de-de','de-en','de-fr','de-pl','en-en','es-es','es-en','es-it','fr-fr','fr-pl','it-it','pl-pl','pl-en','ru-ru','tr-tr','zh-zh','zh-en']
```

The meaning for each abbreviation can be found on the Task article as well. This command will return the Pearson Correlation Score.

## Acknowledgments and References

We acknowledge Prof. Robert Frank for giving us the opportunity of doing this project, and the organization of SemEval 2022 for proposing this task. Besides that, we would like to reference [Nikkei](https://aclanthology.org/2022.semeval-1.171.pdf) and [WueDevils](https://aclanthology.org/2022.semeval-1.175.pdf) from which we got some inspiration.
