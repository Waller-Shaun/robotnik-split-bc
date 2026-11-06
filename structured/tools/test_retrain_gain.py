"""Initialization is independent of frozen-feature cache RNG consumption."""
import unittest
import numpy as np
import torch
from retrain_gain import make_model


class PortableTrainingTests(unittest.TestCase):
    def test_cache_rng_consumption_cannot_change_initial_head(self):
        hidden = np.random.default_rng(20261105).normal(size=(40, 128)).astype(np.float32)
        torch.manual_seed(17)
        torch.randn(200)
        first = make_model(hidden, 'cpu')
        torch.randn(5000)
        second = make_model(hidden, 'cpu')
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)


if __name__ == '__main__':
    unittest.main()
