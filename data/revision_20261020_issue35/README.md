# Updated workbook training snapshot

Imported 570 rows from the supplied workbook's To be Fed sheet. All 551 previous text/label pairs remain unchanged. The user explicitly assigned the 19 additions to training.

Training now has 400 rows: FACT 174, ISSUE 28, RULING 196, OTHER 2. Validation remains 112 rows and test remains 58 rows, with unchanged texts, labels and case assignments. There are 13 cases overall, nine in training. New cases contribute selected ISSUE-labeled examples only, not full annotated opinions.

Recovered case provenance:

- Excel row 553: G.R. L-29330, December 14, 1987, already a training case. https://lawphil.net/judjuris/juri1987/dec1987/gr_l-29330_1987.html
- Rows 554–559: G.R. 225562, March 8, 2017. https://lawphil.net/judjuris/juri2017/mar2017/gr_225562_2017.html
- Rows 560–565: G.R. 145441, April 26, 2005. https://lawphil.net/judjuris/juri2005/apr2005/gr_145441_2005.html
- Rows 566–571: G.R. 141877, August 13, 2004, matched to the previously inspected supplied PDF.

E-Library searches also identified the Louh and PSBank cases, but full-page retrieval timed out. LawPhil was used as the fallback for provenance. The workbook remains the source of training wording and labels; source identification does not certify the labels.

All supplied labels are preserved as provisional. Eleven new rows have NEEDS_DISCUSSION flags because supporting arguments or historical assertions can fall outside ISSUE under the provisional codebook. They are included in this user-requested provisional training experiment, not represented as human-approved.

The new identifiers ending ADD20261020_R plus the Excel row distinguish these workbook records from the earlier GR141877 R1 annotation table. That table is not merged into this dataset, avoiding duplicate examples. No workbook cells were modified during this import.

All examples were tokenized using nlpaueb/legal-bert-base-uncased revision 15b570cbf88259610b082a167dacc190124f60f6, including special tokens without truncation or padding. Maximum length: 296 tokens. Case separation and exact normalized-text uniqueness were checked.

Experiment: eight epochs from the pretrained base, seed 42, learning rate 0.0002, standard loss weights and no oversampling. Model output: models/legalbert_issue35. Test evaluation remains disabled.
