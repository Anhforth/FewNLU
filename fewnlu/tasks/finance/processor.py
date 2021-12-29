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

"""
This file contains the logic for loading training and test data for all SuperGLUE tasks.
"""

import json
import os
import random
from collections import Counter
from typing import List, Dict, Callable

import log
from utils import InputExample
from global_vars import AUGMENTED_SET, TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET
from tasks.base_processor import DataProcessor

logger = log.get_logger()

class FinanceDataProcessor(DataProcessor):
    """
    Data processsor for SuperGLUE tasks.
    """

    TRAIN_FILE = "train.jsonl"
    DEV_FILE = "val.jsonl"
    # DEV32_FILE = "dev32.jsonl"
    TEST_FILE = "test.jsonl"
    UNLABELED_FILE = "unlabeled.jsonl"
    AUGMENTED_FILE = "augmented.jsonl"

    # WSC_TRAIN_FILE_FOR_CLS = "train_for_cls.jsonl"
    # WSC_DEV32_FILE_FOR_CLSF = "dev32_for_cls.jsonl"

    def __init__(self, task_name: str):
        super(FinanceDataProcessor, self).__init__()
        self.task_name = task_name
        assert self.task_name in FINANCE_PROCESSORS

    def get_train_examples(self, data_dir, use_cloze):
        """
        if not use_cloze and self.task_name == "wsc":
            logger.info("Loading CLS train set for WSC task.")
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_TRAIN_FILE_FOR_CLS),  TRAIN_SET, use_cloze=False)
        elif self.task_name=='wsc':
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_TRAIN_FILE_FOR_CLS),  TRAIN_SET, use_cloze=True)
        """
        return self._create_examples(os.path.join(data_dir, FinanceDataProcessor.TRAIN_FILE), TRAIN_SET)

    """
    def get_dev32_examples(self, data_dir, use_cloze):
        if not use_cloze and self.task_name == "wsc":
            logger.info("Loading CLS dev32 set for WSC task.")
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_DEV32_FILE_FOR_CLS),  TRAIN_SET, use_cloze=False)
        elif self.task_name=='wsc':
            return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.WSC_DEV32_FILE_FOR_CLS),  TRAIN_SET, use_cloze=True)
        return self._create_examples(os.path.join(data_dir, SuperGLUEDataProcessor.DEV32_FILE), DEV32_SET)
    """

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, FinanceDataProcessor.DEV_FILE), DEV_SET)

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, FinanceDataProcessor.TEST_FILE), TEST_SET)

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, FinanceDataProcessor.UNLABELED_FILE), UNLABELED_SET)

    def get_augmented_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, FinanceDataProcessor.AUGMENTED_FILE), AUGMENTED_SET)


class EDProcessor(FinanceDataProcessor):
    """Processor for the SuperGLUE RTE task."""

    def get_labels(self):
        return ["0", "1", "2"]

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='GBK') as f:
            for idx,line in enumerate(f):
                example_json = json.loads(line)
                # idx = example_json['idx']
                if isinstance(idx, str):
                    idx = int(idx)
                label = str(example_json.get('label'))
                if str(label) not in self.get_labels():
                    label = "2"
                # label = "T" if example_json.get('label') else "F"
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['text']
                example = InputExample(guid=guid, text_a=text_a, label=label, idx=idx)
                examples.append(example)
        return examples

FINANCE_PROCESSORS = {
    "ed": EDProcessor,
}  # type: Dict[str,Callable[[],FinanceDataProcessor]]

