'''

Timm model classes

Last modified: 07/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: timm_models.py
'''

import torch
import torch.nn as  nn
import torch.nn.functional as F
import timm

from utilities import *

class timmForSpeechClassification(nn.Module):
    """
    Create any number of timm models to train for speech classification
    """
    def __init__(self, name, label_dim, 
                 shared_dense=False, sd_bottleneck=150, clf_bottleneck=150,
                 activation='relu', final_dropout=0.2, layernorm=False):
        """
        :param name: timm model name, e.g. tf_efficientnet_b2_ns (str)
        :param label_dim: specify number of categories to classify - expects either a single number or a list of numbers
        :param shared_dense: specify whether to add a shared dense layer before classification head
        :param sd_bottleneck: size to reduce to in shared dense layer
        :param clf_bottleneck: size to reduce to in intial classifier dense layer
        :param activation: activation function for classification head
        :param final_dropout: amount of dropout to use in classification head
        :param layernorm: include layer normalization in classification head
        """
        super().__init__()

        self.name = name
        self.label_dim = label_dim
        self.sd_bottleneck=sd_bottleneck
        self.clf_bottleneck=clf_bottleneck

        # Use timm
        model = timm.create_model(self.name, pretrained=False, in_chans=1)

        self.clsf = model.default_cfg['classifier']
        self.embedding_dim = model._modules[self.clsf].in_features
        model._modules[self.clsf] = nn.Identity()
        self.model = model

        #adding a shared dense layer
        self.shared_dense = shared_dense
        if self.shared_dense:
            self.dense = nn.Linear(self.embedding_dim, self.sd_bottleneck)
            self.clf_input = self.sd_bottleneck
        else:
            self.clf_input = self.embedding_dim

        self.classifiers = []
        if isinstance(self.label_dim, list):
            for dim in self.label_dim:
                self.classifiers.append(ClassificationHead(input_size=self.clf_input, bottleneck=self.clf_bottleneck, output_size=dim,
                                             activation=activation, final_dropout=final_dropout,layernorm=layernorm))
        else:
            self.classifiers.append(ClassificationHead(input_size=self.clf_input, bottleneck=self.clf_bottleneck, output_size=self.label_dim,
                                             activation=activation, final_dropout=final_dropout,layernorm=layernorm))
            
        self.classifiers = nn.ModuleList(self.classifiers)

    def extract_embedding(self, x, embedding_type = 'ft', pooling_mode='mean'):
        """
        Extract an embedding from various parts of the model
        :param x: waveform input (batch size, input size)
        :param embedding_type: 'ft', 'pt', to indicate whether to extract from classification head (ft), base model (pt), or shared dense layer (st)
        :param pooling_mode: method of pooling embeddings if required ("mean" or "sum")
        :return e: embeddings for a batch (batch_size, embedding dim)
        """
        if embedding_type == 'ft':
            assert pooling_mode == 'mean' or pooling_mode == 'sum', f"Incompatible pooling given: {pooling_mode}. Please give mean or sum"

            activation = {}
            def _get_activation(name):
                def _hook(model, input, output):
                    activation[name] = output.detach()
                return _hook
            
            x = self.model(x)
            if self.shared_dense:
                x = self.dense(x)
            
            embeddings = []
            for clf in self.classifiers:
                clf.head.dense.register_forward_hook(_get_activation('embeddings'))
                logits = clf(x)
                embeddings.append(activation['embeddings'])

            e = activation['embeddings']

            embeddings = torch.stack(embeddings, dim=1)
            if pooling_mode == "mean":
                e = torch.mean(embeddings, dim=1)
            else:
                e = torch.sum(embeddings, dim=1)

        elif embedding_type == 'pt' or embedding_type=='st':
            e = self.model(x)
            if embedding_type == 'st':
                e = self.dense(e)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt) or shared dense (st)')
        return e
    
    def forward(self, x):
        """
        Run model
        :param input_values: input values to the model (batch_size, input_size)
        :return: classifier output (batch_size, num_labels)
        """
        x = self.model(x)
        if self.shared_dense:
            x = self.dense(x)

        preds = []
        for clf in self.classifiers:
            pred = clf(x)
            preds.append(pred)

        logits = torch.column_stack(preds)

        return logits 
    
