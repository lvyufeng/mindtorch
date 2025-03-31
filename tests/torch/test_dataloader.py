import time
import torch
from torch.utils.data import Dataset
import unittest

class TestSlowIndexDataset(Dataset):
    def __init__(self, end: int, slow_index: int):
        self.end = end
        self.slow_index = slow_index

    def __getitem__(self, idx):
        if idx == self.slow_index:
            time.sleep(0.5)
        return idx

    def __len__(self):
        return self.end

class TestOutOfOrderDataLoader(unittest.TestCase):
    def test_in_order_index_ds(self):
        dataset = TestSlowIndexDataset(end=10, slow_index=2)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=2,
            in_order=True,
        )

        expected_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        output = [sample.item() for sample in dataloader]
        self.assertEqual(expected_order, output)
