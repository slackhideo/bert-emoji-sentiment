# Author: Tiago M. de Barros
# Date:   2021-08-07
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

from flask import Flask, request

from classifier import Classifier


app = Flask(__name__)

classifier = Classifier()

@app.route("/classify", methods=["GET", "POST"])
def classify():
    """Classify the sentiment of 'text', either in GET or POST requests."""
    text = request.values.get("text")
    return classifier.classify(text)
