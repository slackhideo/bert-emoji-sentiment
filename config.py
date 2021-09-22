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


################################
# Parameters of the classifier #
################################


# Model to load for classification or training
MODEL = "bertimbau_ttsbr"

# Name of the training file
TRAIN_FILE = "train.csv"

# Name of the development/evaluation file
DEV_FILE   = "dev.csv"

# Name of the test file
TEST_FILE  = "test.csv"

# Size of the batch in number of instances
BATCH_SIZE = 1

# Maximum length of text to consider
MAX_SEQ_LENGTH = 128

# Initial learning rate
LEARNING_RATE = 1e-05

# Number of epochs to train
NUM_TRAIN_EPOCHS = 3

# Path of the model to load
MODEL_DIR = "models/" + MODEL

# Path of the output directory for the trained model
OUTPUT_DIR = "/dev/shm/output/" + MODEL + "_len" + str(MAX_SEQ_LENGTH) + "_bsz" + str(BATCH_SIZE)
