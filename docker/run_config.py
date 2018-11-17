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

import argparse
import flags.flags_parser as flags_parser
import importlib


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file', help='The model file to execute.')
    parser.add_argument('flags_file', help='The flags_file file that is to be executed.')
    # TODO(oskopek): Add parsing of remainder args for overriding flag values.
    parser.add_argument('remainder', nargs=argparse.REMAINDER, help='The remaining command line arguments.')

    args = parser.parse_args()

    flags_parser.parse(args.flags_file, remainder=args.remainder)

    model = importlib.import_module(args.model_file)
    model.run()


if __name__ == "__main__":
    main()
