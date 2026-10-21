import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import core
from transformers import AutoTokenizer
from collections import Counter

# Manually transcribed from the supplied PDF page images; no web substitution.
# page_start, page_end, label, review status, notes, source sentence
rows = [
(3,3,'FACT','PENDING','Family history; issues means children here.',"Macaria Francisco (Macaria) and Marcos Averia contracted marriage which bore six issues, namely: Gregorio, Teresa, Domingo, Angel, Felipe and Felimon."),
(3,3,'FACT','PENDING','Family history; issue means offspring.',"Macaria was widowed and she contracted a second marriage with Roberto Romero (Romero) which bore no issue."),
(3,3,'FACT','PENDING','',"Romero died on February 28, 1968, leaving three adjoining residential lots located at Sampaloc, Manila."),
(3,3,'FACT','PENDING','',"In a Deed of Extrajudicial Partition and Summary Settlement of the Estate of Romero, the house and lot containing 150 square meters at 725 Extremadura Street, Sampaloc was apportioned to Macaria."),
(3,3,'FACT','PENDING','',"Transfer Certificate of Title (TCT) No. 93310 covering the Extremadura property was accordingly issued in the name of Macaria."),
(3,3,'FACT','PENDING','Earlier litigation; not an issue framed for this petition.',"Alleging that fraud was employed by her co-heirs in the partition of the estate of Romero, Macaria filed on June 1, 1970 an action for annulment of title and damages before the Court of First Instance of Manila against her co-heirs Domingo Viray, et al., docketed as Civil Case No. 79955."),
(3,3,'FACT','PENDING','Reports the outcome of an earlier proceeding.',"The case was pending litigation for about ten years until the decision of the Court of Appeals which adjudged Macaria as entitled to an additional 30 square meters of the estate of Romero became final and executory."),
(3,3,'FACT','PENDING','',"Macaria’s son Gregorio and his family and daughter Teresa’s family lived with her at Extremadura until her death on March 28, 1983."),
(3,4,'FACT','PENDING','One complete sentence across the page break.',"Close to six years after Macaria’s demise or on January 19, 1989, her children Domingo, Angel and Felipe, along with Susan Pelayo vda. de Averia (widow of Macaria’s deceased son Felimon), filed before the Regional Trial Court (RTC) of Manila a complaint against their brother Gregorio and niece Sylvanna Vergara “representing her absentee mother” Teresa Averia, for judicial partition of the Extremadura property inclusive of the 30 square meters judicially awarded."),
(4,4,'FACT','PENDING','Factual allegation of a transaction; not a judicial holding.',"Gregorio and Sylvanna further countered that the plaintiff Domingo sold and assigned to the spouses Gregorio and Agripina his one sixth (1/6) share in the remaining ½ portion of the Extremadura property."),
(5,5,'FACT','PENDING','',"During the pendency of the case or on June 7, 1989, Macaria’s son Felipe executed a Waiver-Affidavit waiving his “share” in the property subject of litigation in favor of his co-heirs."),
(6,6,'FACT','PENDING','Quoted RTC disposition reported as procedural history.',"Both parties are hereby ordered to shoulder their respective expenses for attorney’s fees and litigation costs."),
(6,6,'FACT','PENDING','Reports the appellate ruling that the Supreme Court reviews.',"In reversing the trial court, the appellate court, noting that the alleged transfers made by Macaria and Domingo in favor of Gregorio were bereft of any written memoranda, held that it was error for the trial court to rely solely on the evidence adduced by the defendants consisting of the testimonies of Gregorio, Veronica Bautista, Sylvanna Vergara Clutario, Atty. Mario C. R. Domingo, Felimon Dagondon and Gregorio Averia, Jr."),
(7,7,'FACT','PENDING','Quoted CA reasoning subsequently rejected; not the current Supreme Court holding.',"[T]he alleged conveyances purportedly made by Macaria Francisco and plaintiff-appellant Domingo Averia are unenforceable as the requirements under the Statute of Frauds have not been complied with."),
(8,8,'FACT','PENDING','Quoted CA disposition; distinguishes earlier disposition from current RULING.',"The case is remanded to the court a quo which is directed to effect the partition of the subject property or if not possible, sell the entire lot and distribute the proceeds of the sale based on equal shares among the children of the late Macaria Francisco after debts of the said deceased are paid or settled pursuant to Article 1078 of the Civil Code."),
(8,8,'ISSUE','PENDING','First assignment of error in the present Supreme Court petition: alleged sale of Macaria’s half interest.',"THE COURT OF APPEALS (SECOND DIVISION) ERRED IN ITS FINDING THAT THERE WAS NO SALE OF ONE-HALF (½) OF THE DECEASED MACARIA F. AVERIA’S INTEREST AND OWNERSHIP OVER THE SUBJECT PROPERTY IN FAVOR OF PETITIONER GREGORIO F. AVERIA."),
(9,9,'ISSUE','PENDING','Second assignment of error in the present petition: legality of receiving parol evidence of Domingo’s sale.',"THE COURT OF APPEALS (SECOND DIVISION) ERRED IN ITS FINDING THAT THE RECEPTION OF PAROL EVIDENCE TO THE EFFECT THAT RESPONDENT DOMINGO AVERIA HAD ALREADY SOLD HIS ONE SIXTH (1/6) SHARE IN THE SUBJECT PROPERTY IN FAVOR OF PETITIONER GREGORIO AVERIA IS NOT IN ACCORDANCE WITH LAW."),
(9,9,'OTHER','NEEDS_DISCUSSION','Supporting argument for inapplicability of the statute; alternative ISSUE if codebook treats this as independent issue framing, or FACT under party-argument convention.',"With respect to the application by the appellate court of the Statute of Frauds, petitioners contend that the same refers only to purely executory contracts and not to partially or completely executed contracts as in the instant case."),
(9,9,'ISSUE','NEEDS_DISCUSSION','Specifically challenges the CA finding of timely objections; alternative FACT as reported procedural assertion rather than separate issue framing.',"The finding of the CA that the testimonies of petitioners’ witnesses were timely objected to by respondents is not, petitioners insist, borne out in the records of the case except with respect to the testimony of Gregorio."),
(9,9,'RULING','PENDING','Current Supreme Court answer; singular petitioner retained as printed.',"The Court finds for petitioner."),
(10,10,'RULING','PENDING','Current Court evaluates the evidentiary record supporting its answer.',"Indeed, except for the testimony of petitioner Gregorio bearing on the verbal sale to him by Macaria of the property, the testimonies of petitioners’ witnesses Sylvanna Vergara Clutario and Flora Lazaro Rivera bearing on the same matter were not objected to by respondents."),
(10,10,'RULING','PENDING','',"Following Article 1405 of the Civil Code, the contracts which infringed the Statute of Frauds were ratified by the failure to object to the presentation of parol evidence, hence, enforceable."),
(10,10,'RULING','SOURCE_CHECK','Printed respondents appears inconsistent with the RTC ruling favoring petitioners; wording retained without correction.',"Contrary then to the finding of the CA, the admission of parol evidence upon which the trial court anchored its decision in favor of respondents is not irregular and is not foreclosed by Article 1405."),
(11,11,'RULING','PENDING','Rule expressly applied by the current Court.',"In any event, the Statute of Frauds applies only to executory contracts and not to contracts which are either partially or totally performed."),
(11,11,'FACT','NEEDS_DISCUSSION','Reports petitioners’ claimed payment and possession within the Court’s analysis; alternative RULING if treated as adopted factual application.',"In the case at bar, petitioners claimed that there was total performance of the contracts, full payment of the objects thereof having already been made and the vendee Gregorio having, even after Macaria’s death in 1983, continued to occupy the property until and after the filing on January 19, 1989 of the complaint subject of the case at bar as in fact he is still occupying it."),
(11,11,'RULING','PENDING','',"In proving the fact of partial or total performance, oral evidence may be received as what the trial court in the case at bar did."),
(11,11,'RULING','PENDING','Tolentino passage expressly adopted to explain the applicable proof requirement.',"However it is not enough for a party to allege partial performance in order to render the Statute of Frauds inapplicable; such partial performance must be duly proved."),
(11,11,'RULING','PENDING','Continuation of adopted Tolentino reasoning.',"But neither is such party required to establish such partial performance by documentary proof before he could have the opportunity to introduce oral testimony on the transaction."),
(11,11,'RULING','PENDING','Continuation of adopted Tolentino reasoning.',"The partial performance may be proved by either documentary or oral evidence."),
(11,11,'RULING','PENDING','Supreme Court endorses the trial court’s credibility assessment.',"The testimonies of petitioners’ witnesses being credible and straightforward, the trial court did not err in giving them credence."),
(12,12,'RULING','PENDING','Supreme Court conclusion about sufficiency of evidence.',"The testimony of Sylvana Vergara Clutario, daughter of Teresa, in fact was more than sufficient to prove the conveyance of half of the subject property by Macaria to Gregorio."),
(12,12,'OTHER','NEEDS_DISCUSSION','Witness-identification question, not a disputed legal question; alternative FACT if codebook groups all evidentiary examination as FACT.',"Are you the same Sylvana Vergara representing the defendant Teresa Averia in this case?"),
(13,13,'FACT','PENDING','Quoted witness account of the consideration for the conveyance.',"Because of her embarrassment, she told them that one half (½) of the House and Lot will be given to the couple to cover the expenses of the case."),
(14,14,'RULING','PENDING','Current Court explains why testimony is credible.',"Her testimony was against the interest of her mother Teresa whom she represented, her mother being also an heir of Macaria."),
(14,14,'RULING','PENDING','Explains adverse interest as a credibility reason; not an ISSUE merely because it uses a conditional.',"If the transfer by Macaria to Gregorio of ½ of the property is upheld as valid and enforceable, then the share of the other heirs including Sylvanna’s mother would considerably be reduced."),
(14,15,'RULING','PENDING','One sentence across pages; names retained as printed.',"As to the sale of Domingo’s 1/6 share to Gregorio, petitioners were able to establish said transaction by parol evidence, consisting of the testimonies of Gregorio Averia, Jr., Veronica Averia and Felimon Dagondon the presentation of which was, it bears repeating, not objected to."),
(15,15,'RULING','PENDING','Current Court weighs Domingo’s denial against witness testimony.',"Such disclaimer cannot, however, prevail over the categorical, positive statements of petitioners’ above-named witnesses."),
(15,15,'RULING','PENDING','Current Court conclusion that performance was proved.',"They proved the full performance or execution of the contracts as well."),
(15,15,'RULING','PENDING','Substantive Supreme Court disposition.',"WHEREFORE, the petition is hereby GRANTED."),
(15,15,'RULING','PENDING','Substantive Supreme Court disposition.',"The January 31, 2000 Decision of the Court of Appeals in CA-G.R. No. 44704 is hereby SET ASIDE."),
(15,15,'RULING','PENDING','Substantive Supreme Court disposition.',"The case is hereby remanded to the trial court, Branch 31 of the RTC of Manila, for appropriate action, following Section 2 of Rule 69 of the Rules of Civil Procedure."),
]
revision='15b570cbf88259610b082a167dacc190124f60f6'
tok=AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased',revision=revision,local_files_only=True)
case='GR141877_20040813'
header='sentence_id page_start page_end text suggested_label review_status notes case_id final_label reviewer split token_count length_status'.split()
print('\t'.join(header))
counts=[]
assert len(set(r[-1] for r in rows))==len(rows)
assert [(r[0],r[1]) for r in rows]==sorted((r[0],r[1]) for r in rows)
for i,(start,end,label,status,notes,text) in enumerate(rows,1):
    n=len(tok(text,add_special_tokens=True,truncation=False,padding=False)['input_ids'])
    counts.append(n)
    fields=[f'{case}_R1_S{i:03d}',str(start),str(end),text,label,status,notes,case,'','','UNASSIGNED',str(n),'PASS' if n<=512 else 'OVER_LIMIT']
    assert len(fields)==13 and all('\t' not in x and '\n' not in x for x in fields)
    print('\t'.join(fields))
print(json.dumps({'rows':len(rows),'labels':dict(Counter(r[2] for r in rows)),'review_status':dict(Counter(r[3] for r in rows)),'min_tokens':min(counts),'max_tokens':max(counts),'over_limit':sum(n>512 for n in counts),'tokenizer_revision':revision}))
