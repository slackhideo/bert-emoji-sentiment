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

import time
import logging
from typing import Callable, Dict

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertConfig, BertTokenizer, EvalPrediction, TrainingArguments

from bert_emoji import BertEmojiClassification
from input_dataset import InputDataset
from trainer import Trainer


# Parameters of the classifier
MODEL = "bertimbau_ttsbr"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 1
PRETRAINED_MODEL_DIR = "models/" + MODEL
OUTPUT_DIR = "/dev/shm/output/" + MODEL + "_len" + str(MAX_SEQ_LENGTH) + "_bsz" + str(BATCH_SIZE)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR, do_lower_case=False, tokenize_chinese_chars=False)
        self.config    = BertConfig.from_pretrained(PRETRAINED_MODEL_DIR)
        self.model     = BertEmojiClassification.from_pretrained(PRETRAINED_MODEL_DIR, config=self.config)

    def classify(self, text: str):
        input_strings = [text]
        test_dataset = InputDataset(None, input_strings, self.tokenizer, mode="test", max_seq_length=MAX_SEQ_LENGTH)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=BATCH_SIZE,
            no_cuda=False
        )

        #def simple_accuracy(preds, labels):
        #    return (preds == labels).mean()

        def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                preds = np.argmax(p.predictions, axis=1)
                assert len(preds) == len(p.label_ids)
                return {"acc": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds, average="macro")}
            return compute_metrics_fn

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=build_compute_metrics_fn()
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
            logger.info(label)
            logger.info(probabilities)

        logger.info(f"--- Final time: {time.time() - start_time} seconds ---")

        return label


    def classify_multiple(self, texts: str):
        input_strings = []
        input_strings.append(text)
        test_dataset = InputDataset(None, input_strings, self.tokenizer, mode="test", max_seq_length=MAX_SEQ_LENGTH)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_eval_batch_size=BATCH_SIZE,
            no_cuda=False
        )

        #def simple_accuracy(preds, labels):
        #  return (preds == labels).mean()

        def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                preds = np.argmax(p.predictions, axis=1)
                assert len(preds) == len(p.label_ids)
                return {"acc": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds, average="macro")}
            return compute_metrics_fn

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=build_compute_metrics_fn()
        )



        import os
        import json

        start_time = time.time()

        logger.info("*** Test ***")
        pred_result = trainer.predict(test_dataset=test_dataset)
        logger.info(f"--- Predict time: {time.time() - start_time} seconds ---")
        predictions   = pred_result.predictions
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=-1)
        predictions   = np.argmax(predictions, axis=-1)
        label_list    = test_dataset.get_labels()
        output_test_file = os.path.join(OUTPUT_DIR, "classification_results.tsv")
        output_test_json = os.path.join(OUTPUT_DIR, "classification_probabilities.json")

        if trainer.is_world_master():
          logger.info("***** Test results *****")
          logger.info(pred_result.metrics)
          with open(output_test_file, "w") as writer:
            writer.write("index\tprediction\ttext\n")
            for idx, pred in enumerate(predictions):
              pred = label_list[pred]
              writer.write(f"{idx}\t{pred}\t{self.tokenizer.decode(test_dataset[idx].input_ids, True, True)}\n")

          predict_results = []
          for probs in probabilities:
            predict_results.append([float(x) for x in torch.flatten(probs)])

          with open(output_test_json, "w") as fp:
            json.dump(predict_results, fp, indent=4)

        logger.info(f"--- Final time: {time.time() - start_time} seconds ---")

        return predictions
