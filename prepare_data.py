import csv
from transformers import AutoTokenizer
import json
from googletrans import Translator

def prepare_data_bert(filename, dir, translation=False):
    data_prepared = []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        counter = 1
        for row in reader:
            id1, id2 = row['pair_id'].split('_')
            temp_dict = {'lang1': row['url1_lang'], 'lang2': row['url2_lang']}
            try:
                with open(dir+'/'+id1[-2:]+'/'+id1+'.json','r') as f:
                    data = json.load(f)
                    if (data['text'] == ''):
                        continue

                    if (translation == True and row['url1_lang'] != 'en'):
                        translator = Translator()
                        data['text'] = translator.translate(data['text']).text
                        data['title'] = translator.translate(data['title']).text
                        print("Translated "+str(counter))
                        counter += 1
                
                    temp_dict['text1'] = data['title']+'[SEP]'+data['text']

                with open(dir+'/'+id2[-2:]+'/'+id2+'.json','r') as f:
                    data = json.load(f)
                    if (data['text'] == ''):
                        continue

                    if (translation == True and row['url2_lang'] != 'en'):
                        translator = Translator()
                        data['text'] = translator.translate(data['text']).text
                        data['title'] = translator.translate(data['title']).text
                        print("Translated "+str(counter))
                        counter += 1
                        
                    temp_dict['text2'] = data['title']+'[SEP]'+data['text']
                
                temp_dict['overall'] = float(row['Overall'])
                data_prepared.append(temp_dict)
                print("Appended "+str(counter))
                counter += 1

            except:
                pass
    
    return data_prepared