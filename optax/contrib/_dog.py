# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Distance over gradients algorithm and its variants.

References:
  Ivgi et al., `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step
  Size Schedule<https://arxiv.org/abs/2302.12022>`_, 2023.

  Khaled et al., `DoWG Unleashed: An Efficient Universal Parameter-Free
  Gradient Descent Method<https://arxiv.org/pdf/2305.16284>`_, 2023.
"""

from optax._src.dog import dog
from optax._src.dog import DoGState
from optax._src.dog import dowg
from optax._src.dog import DoWGState
from optax._src.dog import scale_by_dog
from optax._src.dog import scale_by_dowg
