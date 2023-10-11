from absl.testing import absltest
from absl.testing import parameterized

import chex
import jax
import jax.numpy as jnp
import numpy as np

from optax.losses import segmentation
from optax.losses import classification

class SoftmaxCrossEntropyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ys = np.array([[2., 1., -2.], [.3, 2., 1.2]], dtype=np.float32)
    self.ts = np.array([[0., 0., 1.], [1., 0., 0.]])
    
  @chex.all_variants
  def test_gamma_zero(self):
    """From gamma == 0 we expect a CE loss."""
    np.testing.assert_allclose(
        self.variant(
            segmentation.sigmoid_focal_loss)(self.ys, self.ts, gamma=0.),
        classification.sigmoid_binary_cross_entropy(self.ys, self.ts),
        atol=1e-4)

