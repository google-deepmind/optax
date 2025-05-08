# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Check all Python files in the optax directory for license header."""

from concurrent import futures
import logging
import pathlib
import pprint
import re
import sys

logger = logging.getLogger(pathlib.Path(__file__).name)
logging.basicConfig(format=logging.BASIC_FORMAT)
logger.setLevel("DEBUG")

# pylint: disable=line-too-long
LICENSE_PATTERN = (
    "(# (pylint|coding).*\n)*"
    "# Copyright 20[0-9][0-9] DeepMind Technologies Limited. All Rights Reserved.\n"
    "#\n"
    "# Licensed under the Apache License, Version 2.0 \\(the \"License\"\\);\n"
    "# you may not use this file except in compliance with the License.\n"
    "# You may obtain a copy of the License at\n"
    "#\n"
    "#     http://www.apache.org/licenses/LICENSE-2.0\n"
    "#\n"
    "# Unless required by applicable law or agreed to in writing, software\n"
    "# distributed under the License is distributed on an \"AS IS\" BASIS,\n"
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
    "# See the License for the specific language governing permissions and\n"
    "# limitations under the License.\n"
    "# ==============================================================================\n"
    ".*"
)
# pylint: enable=line-too-long

LICENSE_TEMPLATE = """
# Copyright 20XX DeepMind Technologies Limited. All Rights Reserved.
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
"""

EXCLUDE_LIST = []


def _check_license_header(fname):
  if fname in EXCLUDE_LIST:
    return True
  try:
    source = pathlib.Path(fname).read_text()
    return re.match(LICENSE_PATTERN, source) is not None
  except UnicodeDecodeError:
    return True

if __name__ == "__main__":
  # check all Python files in the optax directory for license header
  source_files = list(pathlib.Path("./optax").glob("**/*.py"))
  with futures.ThreadPoolExecutor(max_workers=32) as executor:
    results = dict(zip(source_files,
                       executor.map(_check_license_header, source_files)))
  failed_files = [str(fname) for fname, status in results.items() if not status]
  if failed_files:
    logger.error(
        "Files:\n%s\ndon't have the proper license. Please include this license"
        " template at the top of your file:\n%s", pprint.pformat(failed_files),
        LICENSE_TEMPLATE)
    sys.exit(1)  # non-success return
