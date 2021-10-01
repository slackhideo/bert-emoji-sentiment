# bert-emoji-sentiment
Sentiment classification using BERT and emoji.

The neural network in this repository can be used in two main ways:

1. **Inference**: classification of the sentiment of texts.
2. **Fine-tuning**: training using a pre-trained model.

For both, configuration is needed. The relevant documentation is in the **Configuration File** section, in the end of this document.

## 1. Inference

There are two ways to use the neural network to classify the sentiment of texts: through a Flask application (this way it works as an API) or through pure Python code.

### A. Flask Application

The Flask application acts as an API that can be used via HTTP requests (e.g., browser or cURL). It supports GET and POST requests interchangeably (not exactly the best practice, but it is done for convenience).

The code of the application is in the file `sentiment.py`.

How to start the server:

```bash
export FLASK_APP=sentiment
export FLASK_ENV=development
export FLASK_DEBUG=1
flask run -p <PORT>
```

Alternatively, one can source the provided `setup_flask.sh` file, that contains the first three lines above. In this case, the server is started thus:

```bash
source setup_flask.sh
flask run -p <PORT>
```

Play's `application.conf` has an entry `ml.sentimentanalysis.server` to point to this Flask application. The default value is `http://127.0.0.1:9909`.

#### API Endpoints

* `/classify`: Classify the sentiment of a piece of text. It has one parameter `text`, which is a string containing the text to be classified. The return is a string containing the inferred sentiment: `-1` (negative), `0` (neutral), or `1` (positive).
* `/classify_multiple`: Classify the sentiment of multiple texts in a JSON list. It has one parameter `texts`, which is a JSON string of a list containing the texts to be classified. The return is a JSON string of a list containing the inferred sentiment of each piece of text, in the same order as the input JSON string.

Examples of how to call the API:

```bash
curl http://127.0.0.1:9909/classify -d "text=Meu lindo exemplo"
curl http://127.0.0.1:9909/classify_multiple -d 'texts=["Sei lá", "Mamãe é legal"]'
```

### B. Pure Python Code

Instead of calling an API, it is possible to perform classification from Python code. All the inference methods are contained in a class called `Classifier`. The first thing to do is to import it:

```python
from classifier import Classifier
```

Then, one can call the inference methods, which have the same name as their corresponding API endpoints and their parameters, with the exception of `file_path`:

* `classify(text: str)`
* `classify_multiple(texts: str)`

The meaning of the parameters and the return values of the methods are the same as their API counterparts.

Examples of how to use the `Classifier` class:

```python
from classifier import Classifier
classifier = Classifier()
print(classifier.classify("Meu lindo exemplo"))
print(classifier.classify_multiple('["Sei lá", "Mamãe é legal"]'))
```

## 2. **Fine-Tuning**

Fine-tuning is performed via Python code. The easiest way to do that is by executing the file `finetuner.py`, which supports two parameters: `-d` (_directory_ [required], specify directory containing the training files) and `-v` (_verbose_ [optional], enable verbose logging).

The code for fine-tuning expects three input files:

* Training file, containing the labelled training samples. Default name: `train.csv`
* Development/evaluation file, containing the labelled evaluation samples. Default name: `dev.csv`
* Test file, containing the labelled test samples. Default name: `test.csv`

The file names can be changed in the configuration file, although the format of these files must be the following:

| "index" | "label" | "text"          |
| :-----: | :-----: | :-------------: |
| "0"     | "-1"    | [negative text] |
| "1"     | "0"     | [neutral text]  |
| "2"     | "1"     | [positive text] |
| ...     | ...     | ...             |

The files should be tab-separated values, with all fields enclosed by double quotes. The `index` field acts as a unique identifier for each entry. The `label` field denotes the sentiment of the text: `-1` (negative), `0` (neutral), or `1` (positive). And the `text` field is the comment or text itself.

Examples of how to start the fine-tuning:

```bash
python finetuner.py -v -d /path/to/the/inputfiles/
```

If needed, it is possible to start the fine-tuning via Python code:

```python
from finetuner import Finetuner
tuner = Finetuner()
tuner.train("/path/to/the/inputfiles/")
```

## Configuration File

The settings of the neural network are stored in the configuration file `config.py`. Some settings are exclusive for fine-tuning while others affect inference and fine-tuning.

### A. Inference and Fine-Tuning

* `MODEL` and `MODEL_DIR`: The code uses `MODEL_DIR` to locate the pre-trained model to load. By default, `MODEL_DIR` is defined based on `MODEL`, for convenience.
* `BATCH_SIZE`: Size of the batch in number of instances (comments) to be used for inference and fine-tuning.
* `MAX_SEQ_LENGTH`: Maximum length of text to consider. The value for inference cannot exceed the one for fine-tuning.

### B. Fine-Tuning

* `TRAIN_FILE`: Name of the training file.
* `DEV_FILE`: Name of the development/evaluation file.
* `TEST_FILE`: Name of the test file.
* `LEARNING_RATE`: Initial learning rate.
* `NUM_TRAIN_EPOCHS`: Number of epochs to train.
* `OUTPUT_DIR`: Path of the output directory for the fine-tuned model.
