import copy
import unittest
from grounding import finalize, select_evidence
from delivery import prepare_delivery


def result(explicit=False, quoted=False):
    texts = [('FACT', 'The buyer paid part of the price.'),
             ('RULING', 'THE RULING OF THE COURT The petition is denied.')]
    if explicit:
        texts.insert(1, ('ISSUE', 'The issue is whether the sellers may rescind the sale.'))
    if quoted:
        texts.append(('FACT', 'The trial court stated:'))
    texts += [('RULING', 'The petitioners, as successors-in-interest of the vendor, are not the injured parties entitled to a rescission of the deed of absolute sale.'),
              ('RULING', 'The sellers failed to deliver the required title.'),
              ('RULING', 'SO ORDERED.' if quoted else 'The obligation remained unfulfilled.'),
              ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.')]
    rows = [dict(id=i+1, role=role, text=text) for i, (role, text) in enumerate(texts)]
    return {**finalize(rows, {}, select_evidence(rows)), 'sentences': rows}


class SynthesisTests(unittest.TestCase):
    def test_missing_issue_gets_labeled_question_without_changing_raw_result(self):
        original = result()
        before = copy.deepcopy(original)
        issue = prepare_delivery(original)['sections']['ISSUE']
        self.assertEqual(issue['status'], 'source_supported_synthesis')
        self.assertEqual(issue['display_text'], 'Whether the petitioners, as successors-in-interest of the vendor, are entitled to a rescission of the deed of absolute sale.')
        self.assertEqual(len(issue['evidence']), 2)
        self.assertFalse(issue['synthesis']['legal_accuracy_verified'])
        self.assertEqual(original, before)
        self.assertEqual(original['blocks']['ISSUE'], '')

    def test_explicit_question_is_never_replaced(self):
        issue = prepare_delivery(result(explicit=True))['sections']['ISSUE']
        self.assertEqual(issue['status'], 'source_backed_draft')
        self.assertIn('sellers may rescind', issue['display_text'])

    def test_quoted_prior_holding_cannot_supply_question(self):
        self.assertEqual(prepare_delivery(result(quoted=True))['sections']['ISSUE']['status'], 'withheld')

    def test_bad_source_hash_or_missing_audit_cannot_trigger_fallback(self):
        for variant in ('hash', 'audit', 'draft'):
            item = result()
            if variant == 'hash': item['verification']['source_sha256'] = 'wrong'
            elif variant == 'audit': item['verification']['blocks'].pop('ISSUE')
            else: item['blocks']['ISSUE'] = 'Invented question.'
            self.assertEqual(prepare_delivery(item)['sections']['ISSUE']['status'], 'withheld')

    def test_no_adjacent_explanation_or_multiple_conclusions_abstains(self):
        from issue_synthesis import synthesize_issue
        item = result()
        evidence = item['verification']['blocks']['RULING']['evidence']
        conclusion = next(e for e in evidence if 'entitled to' in e['text'])
        self.assertIsNone(synthesize_issue(item['sentences'], [conclusion]))
        rows = item['sentences'] + [dict(id=99, role='RULING', text=conclusion['text']),
                                   dict(id=100, role='RULING', text='Another explanation.')]
        more = [dict(sentence_id=99, text=conclusion['text'], selection='court_reasoning_context'),
                dict(sentence_id=100, text='Another explanation.', selection='court_reasoning_context')]
        self.assertIsNone(synthesize_issue(rows, evidence + more))


if __name__ == '__main__': unittest.main()
