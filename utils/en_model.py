from modulefinder import Module
import torch
from torch import nn
from torch.nn.functional import dropout
from transformers import RobertaModel, HubertModel, Data2VecAudioModel

from utils.cross_attn_encoder import MambaLayer_singlelayer
from utils.cross_attn_encoder import  BertConfig, SAGELayer, MambaLayer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class rob_d2v_cc(nn.Module):            
    def __init__(self, config):        
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
          )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        T_features = raw_output["pooler_output"]  # Shape is [batch_size, 768]

        # audio feature extraction
        # A_hidden_states = self.data2vec_model(audio_inputs, audio_mask).last_hidden_state
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0]!=0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features,0).to(device)


        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_features, A_features), dim=1)    # Shape is [batch_size, 768*2]
        fused_output = self.fused_output_layers(fused_features)        # Shape is [batch_size, 1]

        return {
                'T': T_output, 
                'A': A_output,
                'M': fused_output
        }

class BiDirectionalGatedMechanism(nn.Module):
    def __init__(self, feature_dim, config):
        super(BiDirectionalGatedMechanism, self).__init__()
        self.feature_dim = feature_dim

        self.audio_gate = nn.Linear(feature_dim, feature_dim)
        self.text_gate = nn.Linear(feature_dim, feature_dim)
        self.drop = nn.Dropout(config.dropout)
    def forward(self, audio_features, text_features):
        Gate_audio = torch.sigmoid(self.audio_gate(audio_features))  # [batch_size, feature_dim]
        Gate_audio = self.drop(Gate_audio)
        Gate_text = torch.sigmoid(self.text_gate(text_features))  # [batch_size, feature_dim]
        Gate_text = self.drop(Gate_text)
        weighted_audio = Gate_text * audio_features  # [batch_size, feature_dim]
        weighted_text = Gate_audio * text_features  # [batch_size, feature_dim]
        fused_features = weighted_audio + weighted_text  # [batch_size, feature_dim]

        return fused_features

class rob_d2v_mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.version = config.fuse_version
        self.config = config
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')

        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        if self.version == 'v3':
            self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768 * 2)
            self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768 * 2)
        else:
            self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
            self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        # Mamba layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers)
        self.Mamba_layers = MambaLayer_singlelayer(Bert_config)
        # fusion method V3
        if self.version == 'v4':
            encoder_layer = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=12, batch_first=True)
            self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)
            encoder_layer = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=12, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
            self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
            self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)

        if self.version == 'v2':
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),

                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768 * 2, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        self.bi_gated_model = BiDirectionalGatedMechanism(768, config).to(device)

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'text_mixed':
            embedding_layer = self.text_mixed_cls_emb
        elif layer_name == 'audio_mixed':
            embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        T_hidden_states = raw_output.last_hidden_state
        T_features = raw_output["pooler_output"]  # Shape is [batch_size, 768]
        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
        ## create new audio mask
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1
        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')  # add cls token
        text_inputs, audio_inputs = self.Mamba_layers(text_inputs, text_attn_mask, audio_inputs, audio_attn_mask)
        # different fusion methods
        if self.version == 'v1':
            # fused features
            fused_features = torch.cat((text_inputs[:, 0, :], audio_inputs[:, 0, :]),
                                            dim=1)  # Shape is [batch_size, 768*2]
        elif self.version == 'v2':
            fused_features = self.bi_gated_model(text_inputs[:, 0, :],
                                                 audio_inputs[:, 0, :])  # shape is [batch_size, 768]

        # outputlayer
        fused_output = self.fused_output_layers(fused_features)

        return fused_output