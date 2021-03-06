# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TextClassificationPipeline,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch, slow

from .test_pipelines_common import ANY, PipelineTestCaseMeta


@is_pipeline_test
class TextClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

    @require_torch
    def test_pt_bert_small(self):
        text_classifier = pipeline(
            task="text-classification", model="Narsil/tiny-distilbert-sequence-classification", framework="pt"
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_1", "score": 0.502}])

    @require_tf
    def test_tf_bert_small(self):
        text_classifier = pipeline(
            task="text-classification", model="Narsil/tiny-distilbert-sequence-classification", framework="tf"
        )

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "LABEL_1", "score": 0.502}])

    @slow
    @require_torch
    def test_pt_bert(self):
        text_classifier = pipeline("text-classification")

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 1.0}])
        outputs = text_classifier("This is bad !")
        self.assertEqual(nested_simplify(outputs), [{"label": "NEGATIVE", "score": 1.0}])
        outputs = text_classifier("Birds are a type of animal")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 0.988}])

    @slow
    @require_tf
    def test_tf_bert(self):
        text_classifier = pipeline("text-classification", framework="tf")

        outputs = text_classifier("This is great !")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 1.0}])
        outputs = text_classifier("This is bad !")
        self.assertEqual(nested_simplify(outputs), [{"label": "NEGATIVE", "score": 1.0}])
        outputs = text_classifier("Birds are a type of animal")
        self.assertEqual(nested_simplify(outputs), [{"label": "POSITIVE", "score": 0.988}])

    def run_pipeline_test(self, model, tokenizer):
        text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        # Small inputs because BartTokenizer tiny has maximum position embeddings = 22
        valid_inputs = "HuggingFace is in"
        outputs = text_classifier(valid_inputs)

        self.assertEqual(nested_simplify(outputs), [{"label": ANY(str), "score": ANY(float)}])
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())

        valid_inputs = ["HuggingFace is in ", "Paris is in France"]
        outputs = text_classifier(valid_inputs)
        self.assertEqual(
            nested_simplify(outputs),
            [{"label": ANY(str), "score": ANY(float)}, {"label": ANY(str), "score": ANY(float)}],
        )
        self.assertTrue(outputs[0]["label"] in model.config.id2label.values())
        self.assertTrue(outputs[1]["label"] in model.config.id2label.values())
