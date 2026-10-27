import copy
import unittest

from grounding import finalize, select_evidence, verify_block


def rows(*items):
    return [dict(id=i + 1, text=text, role=role, confidence=.999)
            for i, (role, text) in enumerate(items)]


class GroundingTests(unittest.TestCase):
    def test_quoted_procedural_rule_cannot_swallow_final_disposition(self):
        source = rows(('FACT', 'The lessee stopped paying rent.'),
                      ('ISSUE', 'The issue is whether judgment on the pleadings was proper.'),
                      ('RULING', 'Section 2 of Rule 35 provides: Sec.'),
                      ('RULING', 'A defending party may move for summary judgment.'),
                      ('RULING', 'WHEREFORE , the petition is DENIED for lack of merit.'),
                      ('RULING', 'SO ORDERED.'))
        plan = select_evidence(source)
        self.assertEqual(plan['disposition_ids'], [5, 6])
        self.assertNotIn(4, [e['sentence_id'] for e in plan['evidence']['RULING']])

    def test_quoted_lower_court_order_after_rule_reference_stays_excluded(self):
        source = rows(('FACT', 'The trial court stated: Section 2 of Rule 35 provides:'),
                      ('RULING', 'WHEREFORE, the complaint is dismissed.'),
                      ('RULING', 'SO ORDERED.'),
                      ('FACT', 'The parties then appealed.'))
        self.assertEqual(select_evidence(source)['disposition_ids'], [])

    def test_application_reasons_preserve_support_and_rebuttal(self):
        source = rows(('FACT', 'A fire destroyed the stalls.'),
                      ('ISSUE', 'The issue is whether the fire was fortuitous.'),
                      ('RULING', 'It is established by evidence that employees failed to prevent the fire.'),
                      ('RULING', 'Such circumstances do not support the theory of fortuitous event.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        output = finalize(source, {}, select_evidence(source))
        self.assertIn(source[2]['text'], output['blocks']['RULING'])
        self.assertIn(source[3]['text'], output['blocks']['RULING'])

    def test_party_through_counsel_is_not_current_application(self):
        for text in (
                'Public respondents, through the Solicitor General, maintain that in the present case there was substantial compliance.',
                'Petitioner, through counsel, argues that it is established by evidence that payment was made.',
                'The trial court held that in the present case the complaint failed to establish a right.'):
            source = rows(('ISSUE', 'The issue is whether payment was due.'),
                          ('RULING', text), ('RULING', 'WHEREFORE, the petition is DENIED.'),
                          ('RULING', 'SO ORDERED.'))
            self.assertNotIn(text, finalize(source, {}, select_evidence(source))['blocks']['RULING'])

    def test_quoted_application_is_not_current_reasoning(self):
        source = rows(('ISSUE', 'The issue is whether payment was due.'),
                      ('FACT', 'The lower court stated: “In the present case, the claimant failed to establish payment.'),
                      ('RULING', 'It is established by evidence that the debt remains unpaid.”'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.'))
        ruling = finalize(source, {}, select_evidence(source))['blocks']['RULING']
        self.assertNotIn('remains unpaid', ruling)

    def test_retrospective_issue_discussion_does_not_hide_reasoning(self):
        source = rows(('FACT', 'The claimant requested payment.'),
                      ('ISSUE', 'The issue is whether the complaint states a cause of action.'),
                      ('RULING', 'In the case at bar, the complaint failed to state the source of the right.'),
                      ('RULING', 'It does not matter that the agency overstepped its bounds by not limiting itself to the issue of whether the complaint states a cause of action.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        plan = select_evidence(source)
        self.assertEqual([e['sentence_id'] for e in plan['evidence']['ISSUE']], [2])
        self.assertIn(3, [e['sentence_id'] for e in plan['evidence']['RULING']])

    def test_embedded_question_in_reasoning_does_not_extend_facts(self):
        source = rows(('FACT', 'The City closed the bridge.'),
                      ('ISSUE', 'The single issue is whether the closure was lawful.'),
                      ('RULING', 'This Court finds that the closure was unlawful.'),
                      ('FACT', 'Witnesses testified about the bridge.'),
                      ('ISSUE', 'The Court of Appeals erred when it disregarded evidence, reasoning that the question of whether the City owned the bridge was factual.'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {}, select_evidence(source))
        self.assertNotIn('erred', out['blocks']['ISSUE'])
        self.assertNotIn('Witnesses', out['blocks']['FACT'])
        self.assertIn('This Court finds', out['blocks']['RULING'])

    def test_background_and_both_positions_survive_procedural_repetition(self):
        source = rows(('FACT', 'The petition challenges the appellate order.'),
                      ('FACT', 'A company constructed a gate on the access road.'),
                      ('FACT', 'Petitioners claim the road is private, supported by a certification.'),
                      ('FACT', 'Respondents contend the road is public.'),
                      *[('FACT', 'The court denied the application concerning the road.') for _ in range(12)],
                      ('ISSUE', 'The issue is whether the court properly denied access to the road.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {}, select_evidence(source))
        for phrase in ('constructed a gate', 'Petitioners claim', 'Respondents contend'):
            self.assertIn(phrase,out['blocks']['FACT'])

    def test_statutory_quotation_does_not_swallow_terminal_disposition(self):
        source = rows(('ISSUE', 'The issue is whether payment is due.'),
                      ('RULING', 'Article 1190 of the Civil Code provides: The parties shall return what they received.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        self.assertEqual(select_evidence(source)['disposition_ids'], [3, 4])

    def test_issue_of_whether_and_following_issues_colon(self):
        for text in ('The issue of whether the petition sufficiently states a claim must be resolved.',
                     'The Court must resolve the following issues: (1) whether payment is due; (2) whether interest is payable.'):
            source = rows(('FACT', text), ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
            self.assertEqual(select_evidence(source)['evidence']['ISSUE'][0]['text'], text)

    def test_question_in_named_precedent_is_not_current_issue(self):
        source = rows(('ISSUE', 'But also in Fery, this Court, passing on the question of whether land must be returned, held that title controls.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        self.assertEqual(select_evidence(source)['evidence']['ISSUE'], [])

    def test_complete_inline_quote_does_not_hide_later_disposition(self):
        source = rows(('FACT', 'The RTC stated that the filing was “deficient.”'),
                      ('ISSUE', 'The issue is whether the filing complied.'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.'))
        self.assertEqual(select_evidence(source)['disposition_ids'], [3, 4])

    def test_misclassified_disposition_preserved_without_relabeling(self):
        source = rows(('FACT', 'The City closed the bridge.'),
                      ('ISSUE', 'The issue is whether the City lawfully closed the bridge.'),
                      ('RULING', 'We hold that the closure was unlawful.'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'),
                      ('ISSUE', 'The appealed decision is REVERSED.'),
                      ('FACT', 'The order is made PERMANENT.'),
                      ('RULING', 'SO ORDERED.'))
        before = copy.deepcopy(source)
        plan = select_evidence(source)
        out = finalize(source, {'RULING': 'The appeal remains pending.'}, plan)
        self.assertEqual(source, before)
        self.assertIn('REVERSED.', out['blocks']['RULING'])
        self.assertIn('PERMANENT.', out['blocks']['RULING'])
        self.assertNotIn('pending', out['blocks']['RULING'])
        self.assertEqual(plan['disposition_ids'], [4, 5, 6, 7])

    def test_quoted_lower_court_order_is_not_final_disposition(self):
        source = rows(('FACT', 'The trial court decision read: WHEREFORE, the petition is GRANTED.'),
                      ('RULING', 'SO ORDERED.'),
                      ('FACT', 'An appeal followed.'),
                      ('ISSUE', 'The issue is whether the trial court erred.'),
                      ('RULING', 'We hold that it did.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'),
                      ('RULING', 'SO ORDERED.'))
        plan = select_evidence(source)
        self.assertEqual(plan['disposition_ids'], [6, 7])
        self.assertNotIn('GRANTED', finalize(source, {}, plan)['blocks']['RULING'])

    def test_only_quoted_order_abstains(self):
        source = rows(('FACT', 'The dispositive portion of the trial court decision read:'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.'))
        self.assertEqual(select_evidence(source)['disposition_ids'], [])

    def test_quoted_precedent_self_reference_is_not_current_court(self):
        source = rows(('FACT', 'The issue is whether the claim should succeed.'),
                      ('RULING', 'An earlier decision explained as follows:'),
                      ('RULING', 'We hold that the claimant is entitled to damages.'),
                      ('RULING', '(Citations omitted) The cases are distinguishable.'),
                      ('RULING', 'WHEREFORE, the claim is DENIED.'), ('RULING', 'SO ORDERED.'))
        output = finalize(source, {}, select_evidence(source))
        self.assertNotIn('entitled to damages', output['blocks']['RULING'])

    def test_certificate_closing_does_not_swallow_court_reasons(self):
        source = rows(('ISSUE', 'The issue is whether the title is valid.'),
                      ('FACT', 'The certification read: The title was registered.'),
                      ('FACT', 'Given this 14th day of December 2012 for reference.'),
                      ('RULING', 'This Court finds that the title is valid.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        output = finalize(source, {}, select_evidence(source))
        self.assertIn('This Court finds that the title is valid.', output['blocks']['RULING'])

    def test_party_argument_is_not_promoted_to_holding(self):
        source = rows(('FACT', 'Petitioners contend that this Court finds the project illegal.'),
                      ('ISSUE', 'The issue is whether the order should stand.'),
                      ('RULING', 'Petitioners argue that the order violates a statute.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {'RULING': 'The Court held that the project is illegal.'}, select_evidence(source))
        self.assertNotIn('illegal', out['blocks']['RULING'])
        self.assertEqual(out['verification']['blocks']['RULING']['mode'], 'source_excerpts')

    def test_missing_explicit_issue_is_withheld(self):
        source = rows(('ISSUE', 'The appealed order is reversed.'))
        out = finalize(source, {'ISSUE': 'Whether the order is reversed?'}, select_evidence(source))
        self.assertEqual(out['blocks']['ISSUE'], '')
        self.assertEqual(out['verification']['blocks']['ISSUE']['mode'], 'withheld')

    def test_negation_actor_date_and_place_changes_rejected(self):
        text = 'On May 3, 2019, the Board did not approve the project in Cebu.'
        evidence = [dict(sentence_id=1, text=text)]
        for candidate in (text.replace('did not approve', 'approved'),
                          text.replace('Board', 'Court'), text.replace('2019', '2020'),
                          text.replace('Cebu', 'Boracay')):
            with self.subTest(candidate=candidate):
                checked = verify_block(candidate, evidence, 'FACT')
                self.assertEqual(checked['text'], text)
                self.assertFalse(checked['generation_accepted'])

    def test_literal_evidence_accepted_but_not_called_legal_accuracy(self):
        text = 'The claim was denied.'
        checked = verify_block(text, [dict(sentence_id=1, text=text)], 'RULING')
        self.assertTrue(checked['generation_accepted'])
        self.assertFalse(checked['legal_accuracy_verified'])

    def test_incomplete_generation_and_token_limit_are_visible(self):
        checked = verify_block('The claim was', [dict(sentence_id=1, text='The claim was denied.')],
                               'RULING', generation={'hit_token_limit': True})
        self.assertEqual(checked['mode'], 'source_excerpts')
        self.assertTrue(any('token limit' in reason for reason in checked['reasons']))
        self.assertTrue(any('complete sentence' in reason for reason in checked['reasons']))

    def test_substantive_text_after_order_prevents_false_terminal_match(self):
        source = rows(('RULING', 'WHEREFORE, the petition is granted.'), ('RULING', 'SO ORDERED.'),
                      ('FACT', 'The appellate proceedings continued after that order.'))
        self.assertEqual(select_evidence(source)['disposition_ids'], [])

    def test_missing_disposition_withholds_confident_sounding_reasoning(self):
        source = rows(('ISSUE', 'The issue is whether the permit is valid.'),
                      ('RULING', 'We hold that the permit is valid.'))
        out = finalize(source, {'RULING': source[-1]['text']}, select_evidence(source))
        self.assertEqual(out['blocks']['RULING'], '')
        self.assertEqual(out['reliability']['status'], 'incomplete')
        self.assertIn('RULING', out['reliability']['missing_roles'])

    def test_appended_dissent_cannot_replace_majority_disposition(self):
        source = rows(('FACT', 'The City closed the bridge.'),
                      ('ISSUE', 'The issue is whether the closure was lawful.'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.'),
                      ('OTHER', 'DISSENTING OPINION'),
                      ('RULING', 'We hold that the closure was lawful.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {}, select_evidence(source))
        self.assertIn('GRANTED', out['blocks']['RULING'])
        self.assertNotIn('DENIED', out['blocks']['RULING'])
        self.assertEqual(out['verification']['excluded_separate_opinion_sentence_ids'], [5, 6, 7, 8])

    def test_separate_opinion_alone_does_not_claim_to_be_the_decision(self):
        source = rows(('OTHER', 'SEPARATE CONCURRING OPINION'),
                      ('RULING', 'WHEREFORE, I vote to grant the petition.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {}, select_evidence(source))
        self.assertTrue(all(not block for block in out['blocks'].values()))

    def test_inline_reported_and_quoted_holdings_are_excluded(self):
        for reported in ('The Court of Appeals held that we find the permit valid.',
                         'The RTC stated, “We hold that the permit is valid.”',
                         'According to petitioners, this Court finds the permit valid.'):
            with self.subTest(reported=reported):
                source = rows(('ISSUE', 'The issue is whether the permit was valid.'),
                              ('RULING', reported),
                              ('RULING', 'WHEREFORE, the permit is VOID.'), ('RULING', 'SO ORDERED.'))
                out = finalize(source, {}, select_evidence(source))
                self.assertNotIn(reported, out['blocks']['RULING'])
                self.assertIn('VOID', out['blocks']['RULING'])

    def test_multisentence_quotation_closes_without_so_ordered(self):
        source = rows(('ISSUE', 'The issue is whether the permit is valid.'),
                      ('FACT', 'The RTC stated: “The agency acted lawfully.'),
                      ('RULING', 'We hold that the permit is valid.”'),
                      ('RULING', 'This Court finds that the permit is void.'),
                      ('RULING', 'WHEREFORE, the permit is VOID.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {}, select_evidence(source))
        self.assertNotIn('permit is valid.”', out['blocks']['RULING'])
        self.assertIn('This Court finds that the permit is void.', out['blocks']['RULING'])

    def test_consecutive_enumerated_issues_are_not_dropped(self):
        source = rows(('OTHER', 'The issues are:'),
                      ('ISSUE', '(a) Whether the permit is valid.'),
                      ('FACT', '(b) Whether damages are payable.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        out = finalize(source, {}, select_evidence(source))
        self.assertIn('Whether the permit is valid.', out['blocks']['ISSUE'])
        self.assertIn('Whether damages are payable.', out['blocks']['ISSUE'])


if __name__ == '__main__':
    unittest.main()
