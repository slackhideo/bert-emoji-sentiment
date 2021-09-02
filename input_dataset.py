# Author: Tiago M. de Barros
# Date:   2021-08-06
#
# Copyright 2021 Tiago Barros.
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

import logging
from enum import Enum
from typing import List

import torch
import numpy as np
from transformers import PreTrainedTokenizer

from utils import InputExample, InputFeatures
from input_processor import InputProcessor


logger = logging.getLogger(__name__)


class Split(Enum):
  train = "train"
  dev   = "dev"
  test  = "test"


class InputDataset(torch.utils.data.Dataset):
  """Dataset for the input texts."""

  def __init__(self,
               data_dir:       str,
               input_strings:  List[str],
               tokenizer:      PreTrainedTokenizer,
               mode:           str,
               max_seq_length: int):
    """Get the input texts and build a dataset using their features."""

    if isinstance(mode, str):
      try:
        mode = Split[mode]
      except KeyError:
        raise KeyError("Mode is not a valid split name")

    self.processor = InputProcessor()

    label_list = self.processor.get_labels()
    self.label_list = label_list

    if data_dir is not None and input_strings is not None:
      raise ValueError("You cannot specify both `data_dir` and `input_strings` at the same time")

    # Directory with csv input files
    elif data_dir is not None:
      logger.info(f"Creating features from dataset file at {data_dir}")
      if mode == Split.test:
        examples = self.processor.get_test_examples(data_dir)
      elif mode == Split.dev:
        examples = self.processor.get_dev_examples(data_dir)
      else:
        examples = self.processor.get_train_examples(data_dir)

    # List with input strings. Only for test/prediction.
    elif input_strings is not None:
      examples = self.processor.get_test_examples_from_strings(input_strings)

    else:
      raise ValueError("You have to specify either `data_dir` or `input_strings`")

    self.features = self._convert_examples_to_features(examples, tokenizer, max_seq_length, label_list)


  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    return self.features[idx]

  def get_labels(self):
    return self.label_list


  def _convert_examples_to_features(self,
                                    examples:   List[InputExample],
                                    tokenizer:  PreTrainedTokenizer,
                                    max_length: int,
                                    label_list: List[str]):
    """Convert text input to features using the tokenizer provided."""

    if max_length is None:
      max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
      if example.label is None:
        return None
      return label_map[example.label]

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer([example.text_a for example in examples],
                               padding="max_length",
                               truncation=True,
                               max_length=max_length,
                               return_tensors="pt")

    # Extra inputs
    if hasattr(examples[0], "extra") and examples[0].extra is not None:
      batch_encoding_emoji = tokenizer([example.extra for example in examples],
                                       padding="max_length",
                                       truncation=True,
                                       max_length=max_length,
                                       return_tensors="pt")
      batch_encoding["extra_input_ids"]      = batch_encoding_emoji["input_ids"]
      batch_encoding["extra_attention_mask"] = batch_encoding_emoji["attention_mask"]
      batch_encoding["extra_token_type_ids"] = batch_encoding_emoji["token_type_ids"]

    if hasattr(examples[0], "data_id") and examples[0].data_id is not None:
      batch_encoding["data_id"] = torch.tensor([example.data_id for example in examples])


    features = []

    for i in range(len(examples)):
      inputs = {k: batch_encoding[k][i] for k in batch_encoding}
      feature = InputFeatures(**inputs, label=labels[i])
      features.append(feature)

    for i, example in enumerate(examples[:5]):
      logger.info("*** Example ***")
      logger.info(f"guid: {example.guid}")
      logger.info(f"features: {features[i]}")

    return np.array(features)
