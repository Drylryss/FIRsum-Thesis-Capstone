import unittest
from delivery import trim_procedural_clutter, readable_evidence


def evidence():
    return [dict(sentence_id=i, text=text) for i, text in [
        (1, 'Before this Court is a Petition for Review concerning the appellate decision.'),
        (2, 'The lower court dismissed the complaint.'),
        (3, 'Ana and Bea were co-owners of land.'),
        (4, 'Ana constructed a building.'),
        (5, 'Bea sold the lot.'),
        (10, 'Ana filed a motion to intervene.'),
        (11, 'She then filed a motion to dismiss the complaint.'),
        (12, 'The court granted the motion.'),
        (20, 'Bea refused to deliver title.')]]


class PresentationTests(unittest.TestCase):
    def test_complete_motion_group_removed_without_losing_events_or_outcome(self):
        source = evidence()
        result = trim_procedural_clutter(source, 'Whether the seller must deliver title.')
        self.assertEqual([e['sentence_id'] for e in result], [2, 3, 4, 5, 20])
        self.assertEqual(len(source), 9)

    def test_procedural_issue_preserves_motion_history(self):
        source = evidence()
        for issue in ('Whether dismissal was proper.', 'Whether the motion to intervene was timely.',
                      'Whether the court had jurisdiction.'):
            self.assertEqual(trim_procedural_clutter(source, issue), source)

    def test_incomplete_or_mixed_motion_group_retained(self):
        source = evidence()
        source[7]['text'] = 'The court ordered transfer of title.'
        result = trim_procedural_clutter(source)
        self.assertTrue({10, 11, 12} <= {e['sentence_id'] for e in result})

    def test_ruling_qualification_and_complete_disposition_retained(self):
        source = [dict(sentence_id=1, text='We reverse.'),
                  dict(sentence_id=2, text='The bank is liable unless the depositor was negligent.'),
                  dict(sentence_id=3, text='WHEREFORE, the decision is REVERSED.'),
                  dict(sentence_id=4, text='The damages award is DELETED.'),
                  dict(sentence_id=5, text='SO ORDERED.')]
        result = readable_evidence('RULING', source, {3, 4, 5})
        self.assertEqual([e['sentence_id'] for e in result], [3, 4, 2, 5])


if __name__ == '__main__': unittest.main()
