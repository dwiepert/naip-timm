import torch
import torch.nn as  nn
import torch.nn.functional as F
import timm

from utilities import *

class timmForSpeechClassification(nn.Module):
    """
    Create any number of timm models to train for speech classification
    """
    def __init__(self, name, label_dim, activation='relu', final_dropout=0.2, layernorm=False):
        """
        :parma name: timm model name, e.g. tf_efficientnet_b2_ns (str)
        :param label_dim: specify number of categories to classify
        :param activation: activation function for classification head
        :param final_dropout: amount of dropout to use in classification head
        :param layernorm: include layer normalization in classification head
        """
        super().__init__()

        self.name = name
        self.label_dim = label_dim

        # Use timm
        model = timm.create_model(self.name, pretrained=False, in_chans=1)

        self.clsf = model.default_cfg['classifier']
        self.embedding_dim = model._modules[self.clsf].in_features
        model._modules[self.clsf] = nn.Identity()
        self.model = model

        self.classifier = ClassificationHead(self.embedding_dim,self.label_dim, activation, final_dropout, layernorm)

    def extract_embedding(self, x, embedding_type = 'ft'):
        """
        Extract an embedding from various parts of the model
        :param x: waveform input (batch size, input size)
        :param embedding_type: 'ft', 'pt', to indicate whether to extract from classification head (ft), base model (pt)
        :return e: embeddings for a batch (batch_size, embedding dim)
        """
        if embedding_type == 'ft':
            activation = {}
            def _get_activation(name):
                def _hook(model, input, output):
                    activation[name] = output.detach()
                return _hook
            
            self.classifier.head.dense.register_forward_hook(_get_activation('embeddings'))
            
            logits = self.forward(x)
            e = activation['embeddings']

        elif embedding_type == 'pt':
            e = self.model(x)

        else:
            raise ValueError('Embedding type must be finetune (ft) or pretrain (pt)')

        return e
    
    def forward(self, x):
        """
        Run model
        :param input_values: input values to the model (batch_size, input_size)
        :return: classifier output (batch_size, num_labels)
        """
        x = self.model(x)
        x = self.classifier(x)
        return x  
    
