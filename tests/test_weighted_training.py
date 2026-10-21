import unittest
from types import SimpleNamespace
import torch
from weighted_training import class_weights, WeightedTrainer, oversample_issues


class WeightingTests(unittest.TestCase):
    def test_oversampling_preserves_labels_and_original_rows(self):
        rows = [{'split': 'train', 'label': role, 'text': role} for role in ['FACT', 'ISSUE', 'RULING', 'OTHER']]
        sampled = oversample_issues(rows, 6)
        self.assertEqual(len(sampled), 9)
        self.assertEqual(sum(r['label'] == 'ISSUE' for r in sampled), 6)
        sampled[1]['text'] = 'changed'
        self.assertEqual(rows[1]['text'], 'ISSUE')
        self.assertEqual(sampled[2]['text'], 'ISSUE')

    def test_oversampling_rejects_evaluation_data_and_invalid_factors(self):
        for split in ['validation', 'test', 'exemplar']:
            with self.assertRaises(ValueError):
                oversample_issues([{'split': split, 'label': 'ISSUE'}], 6)
        for factor in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                oversample_issues([], factor)

    def test_train_only_counts_and_cap(self):
        rows = [{'label': role} for role, n in [('FACT',174),('ISSUE',9),('RULING',196),('OTHER',2)] for _ in range(n)]
        weights = class_weights(rows, 'balanced', 20)
        self.assertAlmostEqual(weights[1] / weights[0], 174 / 9)
        self.assertEqual(weights[3], 20)

    def test_rare_class_error_has_greater_loss(self):
        trainer = object.__new__(WeightedTrainer)
        trainer.class_weights = torch.tensor([1., 10., 1., 1.])
        labels = torch.tensor([0, 1])
        # Same number of errors; swapping which class is missed changes weighted loss.
        rare_wrong = lambda **kw: SimpleNamespace(logits=torch.tensor([[5.,0.,0.,0.],[5.,0.,0.,0.]]))
        common_wrong = lambda **kw: SimpleNamespace(logits=torch.tensor([[0.,5.,0.,0.],[0.,5.,0.,0.]]))
        self.assertGreater(trainer.compute_loss(rare_wrong, {'labels':labels}), trainer.compute_loss(common_wrong, {'labels':labels}))

    def test_missing_class_rejected(self):
        with self.assertRaises(ValueError):
            class_weights([{'label':'FACT'}], 'balanced')

    def test_dataset_normalization_preserves_weights_across_batches(self):
        trainer = object.__new__(WeightedTrainer)
        trainer.class_weights = torch.tensor([1., 30., 1., 1.])
        trainer.loss_normalizer = 2.
        model = lambda input_ids: SimpleNamespace(logits=input_ids)
        logits = torch.tensor([[0.,5.,0.,0.],[5.,0.,0.,0.]])
        labels = torch.tensor([0,1])
        together = trainer.compute_loss(model, {'input_ids':logits, 'labels':labels})
        separate = sum(trainer.compute_loss(model, {'input_ids':logits[i:i+1], 'labels':labels[i:i+1]}) for i in range(2)) / 2
        self.assertAlmostEqual(float(together), float(separate), places=5)


if __name__ == '__main__':
    unittest.main()
