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

import json
import time
import logging

import torch
import numpy as np
from transformers import BertConfig, BertTokenizer, TrainingArguments

import config
from bert_emoji import BertEmojiClassification
from input_dataset import InputDataset
from trainer import Trainer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_DIR, do_lower_case=False, tokenize_chinese_chars=False)
        self.config    = BertConfig.from_pretrained(config.MODEL_DIR)
        self.model     = BertEmojiClassification.from_pretrained(config.MODEL_DIR, config=self.config)

    def classify(self, text: str):
        input_strings = [text]
        test_dataset = InputDataset(None, input_strings, self.tokenizer, mode="test", max_seq_length=config.MAX_SEQ_LENGTH)

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            per_device_eval_batch_size=config.BATCH_SIZE,
            no_cuda=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args
        )

        start_time = time.time()

        logger.info("*** Test ***")
        pred_result = trainer.predict(test_dataset=test_dataset)
        logger.info(f"--- Predict time: {time.time() - start_time} seconds ---")
        predictions   = pred_result.predictions
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=-1)
        predictions   = np.argmax(predictions, axis=-1)
        label_list    = test_dataset.get_labels()

        if trainer.is_world_master():
            label = label_list[predictions[0]]
            logger.info("***** Test result *****")
            logger.info("Input: %s", text)
            logger.info("Label: %s", label)
            logger.info("Probs: %s", probabilities)

        logger.info(f"--- Final time: {time.time() - start_time} seconds ---")

        return label


    def classify_multiple(self, texts: str):
        try:
            input_strings = json.loads(texts)
        except TypeError:
            return "Parameter 'texts' must be a valid JSON string!"
        except json.JSONDecodeError:
            return "Invalid input JSON!"

        test_dataset = InputDataset(None, input_strings, self.tokenizer, mode="test", max_seq_length=config.MAX_SEQ_LENGTH)

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            per_device_eval_batch_size=config.BATCH_SIZE,
            no_cuda=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args
        )

        start_time = time.time()

        logger.info("*** Test ***")
        pred_result = trainer.predict(test_dataset=test_dataset)
        logger.info(f"--- Predict time: {time.time() - start_time} seconds ---")
        predictions   = pred_result.predictions
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=-1)
        predictions   = np.argmax(predictions, axis=-1)
        label_list    = test_dataset.get_labels()

        if trainer.is_world_master():
          logger.info("***** Test results *****")
          predict_results = []
          for idx, pred in enumerate(predictions):
            label = label_list[pred]
            predict_results.append(label)
            logger.info("Input: %s", input_strings[idx])
            logger.info("Label: %s", label)
            logger.info("Probs: %s", probabilities[idx])
            logger.info("***************")

        logger.info(f"--- Final time: {time.time() - start_time} seconds ---")

        return json.dumps(predict_results)
