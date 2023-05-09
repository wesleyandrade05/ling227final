from transformers import BertTokenizer, BertModel
from prepare_data import prepare_data_bert
import torch
import csv
import numpy as np
import pickle
import sys

def bert_output(pair, tokenizer, model):
    encoded_input_1 = tokenizer(pair['text1'], return_tensors='pt', padding=True, truncation=True)
    encoded_input_2 = tokenizer(pair['text2'], return_tensors='pt',padding=True, truncation=True)

    output1 = model(**encoded_input_1, output_hidden_states=True)
    output2 = model(**encoded_input_2, output_hidden_states=True)

    dense_vec1 = output1.last_hidden_state
    dense_vec2 = output2.last_hidden_state

    encoded_input_1['attention_mask'] = encoded_input_1['attention_mask'][0]
    encoded_input_2['attention_mask'] = encoded_input_2['attention_mask'][0]

    att_mask1 = encoded_input_1['attention_mask'].unsqueeze(-1).expand(dense_vec1.size()).float()
    att_mask2 = encoded_input_2['attention_mask'].unsqueeze(-1).expand(dense_vec2.size()).float()

    masked1 = dense_vec1 * att_mask1
    masked2 = dense_vec2 * att_mask2

    summed1 = torch.sum(masked1, 1)
    summed2 = torch.sum(masked2, 1)

    summed_mask1 = torch.clamp(att_mask1.sum(1), min=1e-9)
    summed_mask2 = torch.clamp(att_mask2.sum(1), min=1e-9)

    pooled1 = summed1/summed_mask1
    pooled2 = summed2/summed_mask2

    pair['vec1'] = pooled1.detach().numpy()
    pair['vec2'] = pooled2.detach().numpy()

    pair['vec1'] = np.reshape(pair['vec1'],pair['vec1'].size)
    pair['vec2'] = np.reshape(pair['vec2'],pair['vec1'].size)


def bert_embedder(filename, dir, embedname, translation=False):
    if (translation == False):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        data = prepare_data_bert(filename, dir, False)

        counter = 1

        for pair in data:
            bert_output(pair, tokenizer, model)
            print("Embedding "+str(counter)+" done")
            counter += 1

        with open('embedding_'+embedname+'.pkl', 'wb') as f:
            pickle.dump(data, f)

    else:
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        model = BertModel.from_pretrained('bert-large-cased')
        data = prepare_data_bert(filename, dir, True)

        counter = 1

        for pair in data:
            bert_output(pair, tokenizer, model)
            print("Embedding "+str(counter)+" done")
            counter += 1

        with open('translation_embedding_'+embedname+'.pkl', 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':

    args = sys.argv

    if (args[1] == 't_multi'):
        bert_embedder('semeval-2022_task8_train-data_batch_2.csv','final_articles','train',False)

    elif (args[1] == 'e_multi'):
        bert_embedder('final_evaluation_data.csv','eval_articles','eval',False)

    elif (args[1] == 't_trans'):
        bert_embedder('semeval-2022_task8_train-data_batch_2.csv','final_articles','train',True)

    elif (args[1] == 'e_trans'):
        bert_embedder('final_evaluation_data.csv','eval_articles','eval',True)