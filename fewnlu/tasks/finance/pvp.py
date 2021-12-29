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
This file contains the different strategies to patternize data for all SuperGLUE tasks, including
direct concatenation, pattern-verbalizer pairs (PVPs), and ptuning PVPs.
"""


import string
from typing import List

from utils import InputExample, get_verbalization_ids
import log
from tasks.base_pvp import PVP, PVPOutputPattern

logger = log.get_logger()

class EDPVP(PVP):

    _is_multi_token = False

    VERBALIZER = {
        "0": ["负"],
        "1": ["正"],
        "2": ["无"]
    }

    MULTI_VERBALIZER={
        "0": ["负"],
        "1": ["正"],
        "2": ["无"]
    }

    def available_patterns(self):
        if not self.use_cloze:
            return [0]
        elif not self.use_continuous_prompt:
            return [1]
        else:
            return [1]

    def get_parts(self, example: InputExample) -> PVPOutputPattern:
        text_a = self.shortenable(example.text_a)  # premise
        # text_b = self.shortenable(example.text_b.rstrip(string.punctuation))  # hypothesis

        assert self.pattern_id in self.available_patterns()

        if not self.use_cloze:
            return [text_a], []
        else:
            return [text_a, "?", 1, "是", [self.mask_id], "的"], []

    def verbalize(self, label) -> List[str]:
        if not self.use_cloze:
            return []

        return EDPVP.VERBALIZER[label]


FINANCE_PVPS = {
    'ed': EDPVP,
}


FINANCE_METRICS = {
    "ed":       ["acc"],
}
