from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from pipeline import FIRPipeline


class Tensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, device):
        return self

    def __getitem__(self, key):
        return self


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(range(10))

    def decode(self, tokens, **kwargs):
        return 'The petition is denied.'

    def __call__(self, text, **kwargs):
        return {'input_ids': Tensor(20)}


class GenerationTests(unittest.TestCase):
    def test_pdf_cleaned_evidence_reaches_generator_without_rewriting_source(self):
        engine = self.engine()
        seen = {}
        def capture(role, source):
            seen[role] = source
            return source, 1
        engine.generate_block = capture
        tagged = [{'id': 1, 'role': 'FACT', 'text': 'The Board paid 12 pesos. 24'},
                  {'id': 2, 'role': 'ISSUE', 'text': 'The issue is whether payment was due.'},
                  {'id': 3, 'role': 'RULING', 'text': 'WHEREFORE, the petition is DENIED.'},
                  {'id': 4, 'role': 'RULING', 'text': 'SO ORDERED.'}]
        markers = [{'number': '24', 'before': 'paid 12 pesos.', 'after': 'The issue is'}]
        result = engine.run_tagged(tagged, 'synthetic', markers)
        self.assertIn('12 pesos.', seen['FACT'])
        self.assertNotIn('24', seen['FACT'])
        self.assertTrue(result['sentences'][0]['text'].endswith('24'))
        self.assertTrue(result['generation_audit']['FACT']['pdf_cleanup_applied'])
        self.assertEqual(result['generation_audit']['FACT']['source_sentence_ids'], [1])
        self.assertEqual(result['verification']['blocks']['FACT']['evidence'][0]['text'], tagged[0]['text'])

    def engine(self):
        engine = FIRPipeline.__new__(FIRPipeline)
        engine.examples, engine.training_metadata, engine.unlearned_roles, engine.device = [], {}, [], 'cpu'
        engine.gt = Tokenizer()
        engine.torch = SimpleNamespace(inference_mode=nullcontext)
        return engine

    def test_output_cap_retried_and_recorded(self):
        engine = self.engine()
        calls = []

        def generate(**kwargs):
            calls.append(kwargs['max_new_tokens'])
            return Tensor(257 if len(calls) == 1 else 30)

        engine.gen = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024), generate=generate)
        text, count = engine.generate_block('RULING', 'The petition is denied.')
        self.assertEqual(calls, [256, 384])
        self.assertFalse(engine.generation_audit['RULING']['hit_token_limit'])
        self.assertTrue(engine.generation_audit['RULING']['calls'][0]['hit_token_limit'])
        self.assertEqual(count, 1)
        self.assertTrue(text.endswith('.'))
        audit = engine.generation_audit['RULING']
        self.assertEqual(audit['source_tokens'], 10)
        self.assertFalse(audit['input_truncated'])
        self.assertEqual(audit['calls'][0]['stage'], 'source')
        self.assertEqual(audit['calls'][0]['input_text'], 'The petition is denied.')

    def test_experiment_does_not_change_default_decoding(self):
        engine = self.engine()
        seen = []
        engine.gen = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024),
                                     generate=lambda **kw: (seen.append(kw) or Tensor(30)))
        engine.generate_block('FACT', 'Source.')
        self.assertNotIn('length_penalty', seen[0])
        engine.generation_overrides = {'length_penalty': 0.8}
        engine.generate_block('FACT', 'Source.')
        self.assertEqual(seen[1]['length_penalty'], 0.8)
        self.assertFalse(seen[1]['do_sample'])

    def test_repeated_cap_is_not_hidden_by_forced_eos(self):
        engine = self.engine()
        engine.gen = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024),
                                     generate=lambda **kw: Tensor(kw['max_new_tokens'] + 1))
        engine.generate_block('RULING', 'The petition is denied.')
        self.assertTrue(engine.generation_audit['RULING']['hit_token_limit'])
        self.assertEqual(len(engine.generation_audit['RULING']['calls']), 2)

    def test_generation_failure_keeps_evidence_and_records_failure(self):
        engine = self.engine()

        def fail(role, source):
            raise RuntimeError('Simulated decoder failure')

        engine.generate_block = fail
        tagged = [dict(id=1, text='WHEREFORE, the petition is denied.', role='ISSUE'),
                  dict(id=2, text='SO ORDERED.', role='RULING')]
        result = engine.run_tagged(tagged, 'synthetic')
        self.assertIn('petition is denied', result['blocks']['RULING'])
        self.assertEqual(result['generated_blocks']['RULING'], '')
        self.assertIn('Simulated decoder failure', result['generation_audit']['RULING']['error'])
        self.assertEqual(result['sentences'][0]['role'], 'ISSUE')


if __name__ == '__main__':
    unittest.main()
