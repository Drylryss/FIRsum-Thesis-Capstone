import fs from 'node:fs/promises';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';
const dir = 'C:/Users/drylr/OneDrive/Documents/Thesis Capstone/outputs/add_gr141877';
const source='C:/Users/drylr/Downloads/Consolidated Tagged Sentences File for LegalBERT.xlsx';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(source));
if(process.argv.includes('--inspect')) {
 console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:1800,tableMaxRows:2,tableMaxCols:2})).ndjson);
 await fs.writeFile(dir+'/before.png',new Uint8Array(await (await wb.render({sheetName:'To be Fed',range:'A1:B4',scale:1})).arrayBuffer()));
} else {
 const data=JSON.parse(await fs.readFile(dir+'/annotations.json','utf8'));
 const s=wb.worksheets.add('GR141877 Review');
 s.showGridLines=false;
 s.getRange('A1:M42').values=data;
 s.getRange('A1:M42').format.font={name:'Arial',size:11};
 s.getRange('A1:M42').format.verticalAlignment='top';
 s.getRange('A1:M42').format.wrapText=true;
 const widths=[36,12,12,100,20,25,70,27,18,22,18,14,18];
 widths.forEach((w,i)=>s.getRangeByIndexes(0,i,42,1).format.columnWidth=w);
 s.getRange('A1:M1').format.fill='#244062';
 s.getRange('A1:M1').format.font={name:'Arial',size:11,bold:true,color:'#FFFFFF'};
 s.getRange('A1:M1').format.rowHeight=32;
 for(let i=1;i<data.length;i++) {
   const lines=Math.max(Math.ceil(data[i][3].length/95),Math.ceil(data[i][6].length/65),2);
   s.getRangeByIndexes(i,0,1,13).format.rowHeight=lines*17+12;
 }
 s.getRange('I2:J42').format.fill='#FFF2CC';
 s.getRange('I2:I42').dataValidation={rule:{type:'list',values:['FACT','ISSUE','RULING','OTHER']}};
 s.getRange('K2:K42').dataValidation={rule:{type:'list',values:['TRAIN','VALIDATION','TEST','RESERVE','UNASSIGNED']}};
 s.getRange('F2:F42').conditionalFormats.add('containsText',{text:'NEEDS_DISCUSSION',format:{fill:'#FFF2CC'}});
 s.getRange('F2:F42').conditionalFormats.add('containsText',{text:'SOURCE_CHECK',format:{fill:'#FCE4D6'}});
 s.freezePanes.freezeRows(1);
 s.getRange('A44:D45').values=[['Source','Supplied Supreme Court E-Library PDF, G.R. No. 141877, August 13, 2004; physical pages 3–15.',null,null],['Review','41 provisional annotations. Final label and reviewer are editable; split remains UNASSIGNED.',null,null]];
 s.getRange('B44:D44').merge(); s.getRange('B45:D45').merge();
 s.getRange('A44:D45').format.wrapText=true;
 s.getRange('A44:D45').format.rowHeight=30;
 wb.recalculate();
 for(const [name,range] of [['review','A15:G21'],['review_end','H36:M42']])
  await fs.writeFile(dir+'/'+name+'.png',new Uint8Array(await (await wb.render({sheetName:s.name,range,scale:1})).arrayBuffer()));
 await (await SpreadsheetFile.exportXlsx(wb)).save(dir+'/Consolidated Tagged Sentences File for LegalBERT.xlsx');
 console.log('Saved workbook with 41 annotations on GR141877 Review.');
}
