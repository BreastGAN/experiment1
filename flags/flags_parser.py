# Copyright 2018 Lukas Jendele and Ondrej Skopek. All Rights Reserved.
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
# ==============================================================================

import json
import re
from dotmap import DotMap

FLAGS = None


def parse(file_name, remainder=None):
    lines = None
    with open(file_name, mode='r') as f:
        lines = f.readlines()

    input_str = re.sub(r'\\\n', '', "\n".join(lines))
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)

    global FLAGS
    FLAGS = DotMap(data)
