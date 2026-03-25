import os
from cProfile import label

import torch
from numpy import dtype
from torch import nn
import transformers
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader


import soundfile as sf
import pandas as pd
import numpy as np
import string
import random



class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, csv_path, audio_directory, mode):
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()

        # store labels
        self.targets = df['label']
        
        # store texts
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
        # store audio
        self.audio_file_paths = []

        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i])
            for j in range(4-len(clip_id)):
                clip_id = '0'+clip_id
            file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
      
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   
        
        
    def __getitem__(self, index):
        #extract text features
        text = str(self.texts[index])
        sound, samplerate = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        tokenized_text = self.tokenizer(
            text,            
            max_length = 64,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )               
                
        # extract audio features    

        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=32000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
            
        return { # text
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                 # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "target":  self.targets[index]
                }
    
    def __len__(self):
        return len(self.targets)


class Dataset_mosi(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length
    
    def __init__(self, csv_path, audio_directory, mode):
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].sort_values(by=['video_id','clip_id']).reset_index()

        # store labels
        self.targets_M = df['label']
        
        # store texts
        df['text'] = df['text'].str[0]+df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # store audio
        self.audio_file_paths = []
        self.audio_file_aguments_output_path = []
        ## loop through the csv entries
        for i in range(0,len(df)):
            file_name = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        # store context
        self.video_id = df['video_id']


        
    def __getitem__(self, index):
        text = str(self.texts[index])
        sound, sample_rate = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

            # tokenize text
        tokenized_text = self.tokenizer(
                text,            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            )
         # extract audio features
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=32000, return_attention_mask=True, truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),

                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,

                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                }

    def __len__(self):
        return len(self.targets_M)
    
    
def collate_fn_sims(batch):   
    text_tokens = []  
    text_masks = []
    audio_inputs = []  
    audio_masks = []
    
    targets = []

   
    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        #audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])

       # labels
        targets.append(batch[i]['target'])

    return {
            # text
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "text_masks": torch.tensor(text_masks, dtype=torch.long),           
            # audio
            "audio_inputs": torch.stack(audio_inputs),
            "audio_masks": torch.stack(audio_masks),
            # labels
            "targets": torch.tensor(targets, dtype=torch.float32),
            }   


def data_loader(batch_size, dataset):
    if dataset == 'mosi':
        csv_path = ''
        audio_file_path = ""
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train')
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test')
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    elif dataset == 'mosei':
        csv_path = ''
        audio_file_path = ""
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train')
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test')
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    else:
        csv_path = ''
        audio_file_path = ""
        train_data = Dataset_sims(csv_path, audio_file_path, 'train')
        test_data = Dataset_sims(csv_path, audio_file_path, 'test')
        val_data = Dataset_sims(csv_path, audio_file_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        return train_loader, test_loader, val_loader
