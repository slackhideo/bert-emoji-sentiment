# Author: Tiago M. de Barros
# Date:   2021-08-17
# Modified from `modeling_bert.py` from Transformers version 3.0.2 by HuggingFace.
#
# Copyright 2021 Tiago Barros.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel


class BertEmojiClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        extra=None,
        extra_input_ids=None,
        extra_attention_mask=None,
        extra_token_type_ids=None,
        classes_weights=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs1 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output1 = outputs1[1]

        outputs2 = self.bert(
            extra_input_ids,
            attention_mask=extra_attention_mask,
            token_type_ids=extra_token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        pooled_output2 = outputs2[1]

        pooled_output = self.dropout(torch.cat((pooled_output1, pooled_output2), dim=1))
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs1[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=classes_weights if classes_weights is not None else None)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
