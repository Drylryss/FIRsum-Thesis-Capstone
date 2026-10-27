import unittest
from footnote_display import display_sentences, pdf_markers


class FootnoteDisplayTests(unittest.TestCase):
    def test_page_end_reference_removed_but_ordinary_number_preserved(self):
        rows=[{'id':1,'text':'The application was denied. 24 Only the intervenors appealed on May 24.'}]
        marker={'number':'24','before':'application was denied.','after':''}
        self.assertEqual(display_sentences(rows,[marker])[1],
                         'The application was denied. Only the intervenors appealed on May 24.')
        marker['before']='denied.'
        self.assertEqual(display_sentences(rows,[marker])[1],rows[0]['text'])

    def test_pdf_geometry_confirms_heading_and_last_run_reference(self):
        from types import SimpleNamespace
        class Page:
            mediabox=SimpleNamespace(width=600)
            def get(self,key,default):return default
            def extract_text(self,visitor_operand_before,visitor_text):
                for text,x,y,color in [('A completed paragraph.',70,600,(0,0,0)),
                                       ('II',295,570,(0,0,0)),
                                       ('This Court holds otherwise.',70,540,(0,0,0)),
                                       ('24',200,546,(1,0,0))]:
                    visitor_operand_before(b'rg',color,None,None)
                    visitor_text(text,[1,0,0,1,0,0],[1,0,0,1,x,y],None,10)
        markers=pdf_markers(Page())
        self.assertEqual([(m['number'],m['kind']) for m in markers], [('II','section_heading'),('24','footnote')])

    def test_inline_roman_numerals_and_unconfirmed_heading_remain(self):
        rows=[{'id':1,'text':'Article II applies to Officer Corliss II. II This Court finds otherwise.'}]
        self.assertEqual(display_sentences(rows,[])[1],rows[0]['text'])
        markers=[{'number':'II','before':'Officer Corliss II.','after':'This Court finds'}]
        self.assertEqual(display_sentences(rows,markers)[1], 'Article II applies to Officer Corliss II. This Court finds otherwise.')

    def test_only_identified_occurrence_removed(self):
        rows=[{'id':1,'text':'The order 3 concerns Road Lot No. 3 on May 3, 2014 for 300 pesos.'}]
        markers=[{'number':'3','before':'The order','after':'concerns Road Lot'}]
        text=display_sentences(rows,markers)[1]
        self.assertEqual(text,'The order concerns Road Lot No. 3 on May 3, 2014 for 300 pesos.')
        self.assertIn('order 3',rows[0]['text'])

    def test_marker_across_sentence_boundary(self):
        rows=[{'id':1,'text':'The claim failed.'},{'id':2,'text':'12 The appeal followed.'}]
        self.assertEqual(display_sentences(rows,[{'number':'12','before':'The claim failed.','after':'The appeal followed.'}])[2],'The appeal followed.')

    def test_unconfirmed_and_ambiguous_numbers_remain(self):
        rows=[{'id':1,'text':'The order 3 applies. The order 3 applies.'}]
        for markers in ([],[{'number':'3','before':'The order','after':'applies.'}]):
            self.assertEqual(display_sentences(rows,markers)[1],rows[0]['text'])
