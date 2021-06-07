import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class classifier(nn.Module):
    def __init__(self, feature_dim, use_dropout=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.Dropout(0.3) if use_dropout else nn.Sequential(),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class AutoModel(nn.Module):
    def __init__(self, pretrained_model=None, use_dropout=False, max_pool=False):
        super().__init__()
        self.model = transformers.AutoModel.from_pretrained(pretrained_model)
        self.feature_dim = self.model.config.hidden_size
        self.classifier = classifier(
            feature_dim=self.feature_dim, use_dropout=use_dropout
        )
        self.max_pool = max_pool

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        x = self.classifier(x)
        x = x.squeeze(-1)
        return x


class AutoModelForClassification(nn.Module):
    def __init__(self, pretrained_model=None, num_labels=1, freeze_backbone=False):
        super().__init__()
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=num_labels
        )
        if freeze_backbone:
            self.freeze()

    def freeze(self):
        for param in self.model.roberta.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.logits
        x = x.squeeze(-1)
        return x


# class bert_base(nn.Module):
#     def __init__(self, use_dropout=False, freeze=False, max_pool=False):
#         super().__init__()
#         self.model = self.get_model()
#         if freeze:
#             self.freeze
#         self.feature_dim = self.model.config.hidden_size
#         self.classifier = classifier(feature_dim=self.feature_dim, use_dropout=use_dropout)
#         self.max_pool = max_pool

#     def forward(self, input_ids, attention_mask):
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         x = outputs.last_hidden_state
#         x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
#         x = self.classifier(x)
#         x = x.squeeze(-1)
#         return x

#     def freeze(self):
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def get_model(self):
#         return transformers.BertModel.from_pretrained("bert-base-uncased")

# class xlnet_base(bert_base):
#     def __init__(self, use_dropout=False, freeze=False, max_pool=False):
#         super(xlnet_base, self).__init__(use_dropout=use_dropout, freeze=freeze, max_pool=max_pool)

#     def get_model(self):
#         return transformers.XLNetModel.from_pretrained("xlnet-base-cased")

# class roberta_base(bert_base):
#     def __init__(self, use_dropout=False, freeze=False, max_pool=False):
#         super(roberta_base, self).__init__(use_dropout=use_dropout, freeze=freeze, max_pool=max_pool)

#     def get_model(self):
#         return transformers.RobertaModel.from_pretrained("roberta-base")
