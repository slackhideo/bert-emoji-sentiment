# Author: Tiago M. de Barros
# Date:   2021-08-05
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

import os
import csv
from typing import List, Tuple

from transformers import DataProcessor, set_seed

from utils import InputExample


process_emoji = True

set_seed(42)


class InputProcessor(DataProcessor):
  """Processor for the input texts."""

  def get_test_examples(self, data_dir):
    """Get a collection of `InputExample`s for prediction."""
    return self._create_examples(os.path.join(data_dir, "emoji100.csv"), "test")

  def get_test_examples_from_strings(self, input_strings: List[str]) -> List[InputExample]:
    """Get a collection of `InputExample`s for prediction."""
    return self._create_test_examples_from_strings(input_strings)

  def get_labels(self):
    """Get the list of labels for this data set."""
    return ["-1", "0", "1"]

  def _create_examples(self, input_file, set_type):
    """Create examples for the training, dev, and test sets."""
    examples = []
    with open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_ALL)
      next(reader, None)
      for i, line in enumerate(reader):
        guid="{0}-{1}".format(set_type, str(i))
        text_a = line[2].replace('""', '"').replace('\\"', '"')
        label = line[1]
        extra = None
        data_id = None #int(line[0])

        if process_emoji:
          text_a, extra = self._split_text_emoji(text_a)
          print(text_a, "|", extra)
          examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, extra=extra, data_id=data_id, label=label))
        else:
          examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, data_id=data_id, label=label))

    return examples


  def _create_test_examples_from_strings(self, input_strings: List[str]) -> List[InputExample]:
    """Create examples for the test set."""

    examples = []

    for i, text in enumerate(input_strings):
      guid    = "{0}-{1}".format("test", str(i))
      text_a  = text.replace('""', '"').replace('\\"', '"') if text else ""
      label   = None
      extra   = None
      data_id = None

      if process_emoji:
        text_a, extra = self._split_text_emoji(text_a)
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, extra=extra, data_id=data_id, label=label))
      else:
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, data_id=data_id, label=label))

    return examples


  def _split_text_emoji(self, text: str) -> Tuple[str, str]:
    """Split emoji from text."""

    # List of emoticons to consider
    emoticons = [":(", "=(", ";(", ":-(", ";-(", ":)", "=)", ";)", ":-)", ";-)", ":D", ";D", "<3", "S2"]

    out_text = text
    emoji_list = []

    # Emoticons
    for e in emoticons:
      idx = 0
      while True:
        idx = out_text.find(e, idx)
        if idx == -1:
          break
        emoji_list.append(out_text[idx:(idx + len(e))])
        out_text = out_text[0:idx] + out_text[(idx + len(e)):]

    # Emoji
    idx = 0
    while idx < len(out_text):
      if ord(out_text[idx]) > 9000:
        emoji_list.append(out_text[idx:(idx + 1)])
        out_text = out_text[0:idx] + out_text[(idx + 1):]
      else:
        idx += 1

    return out_text.strip(), " ".join(emoji_list)
