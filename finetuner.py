# Author: Tiago M. de Barros
# Date:   2021-09-09
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
import json
import logging
import argparse
from typing import Callable, Dict

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertConfig, BertTokenizer, EvalPrediction, TrainingArguments

import config
from bert_emoji import BertEmojiClassification
from input_dataset import InputDataset
from trainer import Trainer


logger = logging.getLogger(__name__)


def main():
    description = "Program to finetune a neural network."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", "--dir",     help="Directory containing the training files")
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    args = parser.parse_args()

    if not args.dir:
        parser.print_help()
        return

    if args.verbose:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO
        )

    finetuner = Finetuner()
    finetuner.train(args.dir)


class Finetuner:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_DIR, do_lower_case=False, tokenize_chinese_chars=False)
        self.config    = BertConfig.from_pretrained(config.MODEL_DIR)
        self.model     = BertEmojiClassification.from_pretrained(config.MODEL_DIR, config=self.config)

    def train(self, data_dir: str):
        train_dataset = InputDataset(data_dir, None, self.tokenizer, mode="train", max_seq_length=config.MAX_SEQ_LENGTH)
        eval_dataset  = InputDataset(data_dir, None, self.tokenizer, mode="dev",   max_seq_length=config.MAX_SEQ_LENGTH)
        test_dataset  = InputDataset(data_dir, None, self.tokenizer, mode="test",  max_seq_length=config.MAX_SEQ_LENGTH)

        epoch_size = len(torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE))

        training_args = TrainingArguments(
            output_dir                  = config.OUTPUT_DIR,
            evaluate_during_training    = True,
            num_train_epochs            = config.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size = config.BATCH_SIZE,
            per_device_eval_batch_size  = config.BATCH_SIZE,
            learning_rate               = config.LEARNING_RATE,
            warmup_steps                = 0,
            weight_decay                = 1e-02,
            logging_dir                 = "logs",
            logging_first_step          = True,
            logging_steps               = epoch_size,
            save_steps                  = epoch_size,
           #save_total_limit            = 25,
            seed                        = 42,
            eval_steps                  = epoch_size,
            no_cuda                     = False
        )

        def build_simple_accuracy_fn() -> Callable[[EvalPrediction], Dict]:
            def simple_accuracy_fn(p: EvalPrediction):
                preds = np.argmax(p.predictions, axis=1)
                assert len(preds) == len(p.label_ids)
                return {"acc": (preds == p.label_ids).mean()}
            return simple_accuracy_fn

        def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                preds = np.argmax(p.predictions, axis=1)
                assert len(preds) == len(p.label_ids)
                return {"acc": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds, average="macro")}
            return compute_metrics_fn

        trainer = Trainer(
            model                = self.model,
            args                 = training_args,
            train_dataset        = train_dataset,
            eval_dataset         = eval_dataset,
            compute_metrics      = build_simple_accuracy_fn(),
           #compute_metrics      = build_compute_metrics_fn(),
            with_classes_weights = True
        )

        # Perform the training
        trainer.train()

        # Save the results for the test dataset
        logger.info("*** Test ***")
        pred_result   = trainer.predict(test_dataset=test_dataset)
        predictions   = pred_result.predictions
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=-1)
        predictions   = np.argmax(predictions, axis=-1)
        label_list    = test_dataset.get_labels()
        output_test_file = os.path.join(config.OUTPUT_DIR, "test_results.tsv")
        output_test_json = os.path.join(config.OUTPUT_DIR, "test_results.probabilities.json")

        if trainer.is_world_master():
          logger.info("***** Test results *****")
          logger.info(pred_result.metrics)
          with open(output_test_file, "w") as writer:
            writer.write("index\tprediction\ttruth\ttext\n")
            for idx, pred in enumerate(predictions):
              pred = label_list[pred]
              writer.write(f"{idx}\t{pred}\t{label_list[pred_result.label_ids[idx]]}\t{self.tokenizer.decode(test_dataset[idx].input_ids, True, True)}\n")

          predict_results = []
          for probs in probabilities:
            predict_results.append([float(x) for x in torch.flatten(probs)])

          with open(output_test_json, "w") as fp:
            json.dump(predict_results, fp, indent=4)


        # Save the resulting probabilities for the training dataset
        logger.info("*** Test Train ***")
        pred_result   = trainer.predict(test_dataset=train_dataset)
        predictions   = pred_result.predictions
        probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions), dim=-1)
        output_train_json = os.path.join(config.OUTPUT_DIR, "train.probabilities.json")

        if trainer.is_world_master():
          logger.info("***** Test results *****")
          logger.info(pred_result.metrics)

          predict_results = []
          for probs in probabilities:
            predict_results.append([float(x) for x in torch.flatten(probs)])

          with open(output_train_json, "w") as fp:
            json.dump(predict_results, fp, indent=4)


        # Save the trained model
        trainer.save_model()
        if trainer.is_world_master():
              self.tokenizer.save_pretrained(config.OUTPUT_DIR)

        return "OK"


if __name__ == "__main__":
    main()
