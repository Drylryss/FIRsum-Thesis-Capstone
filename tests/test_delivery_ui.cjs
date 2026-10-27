// Unit test the shipped page script with an in-memory DOM, without a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.textContent='';this.classList={toggle(){}};this.dataset={};}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;}
 setAttribute(){}
 click(){}
}
const elements=new Map();
const document={createElement:tag=>new Element(tag),querySelectorAll:()=>[],getElementById:id=>{
 if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
}};
const script=fs.readFileSync('web/index.html','utf8').split('<script>')[1].split('</script>')[0];
const ctx={document,fetch:()=>Promise.resolve({json:()=>Promise.resolve({})}),setTimeout(){},URL:{},Blob:class{},navigator:{}};
vm.createContext(ctx);vm.runInContext(script,ctx);
const body={case_id:'Test',sentences:[],blocks:{FACT:'UNSAFE LEGACY FALLBACK'},generated_blocks:{FACT:'RAW FABRICATION'},chunks:{},verification:{blocks:{}}};
ctx.body=body;
assert(!vm.runInContext('copyText(body)',ctx).includes('UNSAFE'));
vm.runInContext('renderOutputs(body)',ctx);
function allText(el){return el.textContent+' '+el.children.map(allText).join(' ');}
assert(!allText(elements.get('digest-preview')).includes('UNSAFE'));
assert(!allText(elements.get('digest-preview')).includes('RAW FABRICATION'));
assert(allText(elements.get('pegasus-output')).includes('REJECTED / UNVERIFIED'));
const rawSection=elements.get('pegasus-output').children[0];
assert(rawSection.children.some(c=>c.tag==='details' && !c.open));
body.delivery={case_id:'Test',notice:'Review required',warnings:[],sections:Object.fromEntries(['FACT','ISSUE','RULING'].map(role=>[role,{status:'source_backed_draft',evidence:[{sentence_id:1,text:'SOURCE ONLY'}]}]))};
assert(vm.runInContext('copyText(body)',ctx).includes('SOURCE ONLY'));
assert(!vm.runInContext('copyText(body)',ctx).includes('RAW FABRICATION'));
let saved;
ctx.capture=(payload,filename)=>{saved={payload,filename};};
vm.runInContext('result=body; saveJSON=capture;',ctx);
elements.get('download').onclick();
assert.strictEqual(saved.payload.sections.FACT.text,'SOURCE ONLY');
assert.strictEqual(saved.payload.sections.FACT.evidence,body.delivery.sections.FACT.evidence);
assert(!JSON.stringify(saved.payload).includes('RAW FABRICATION'));
elements.get('download-pegasus').onclick();
assert.strictEqual(saved.payload.artifact_type,'diagnostic_only_not_a_digest');
body.delivery.sections.ISSUE={status:'source_supported_synthesis',display_text:'Whether payment is due.',evidence:[{sentence_id:1,text:'Payment is not due.'}],synthesis:{scope:'One inferred question.'}};
assert(vm.runInContext('copyText(body)',ctx).includes('Source-supported synthesis — not verbatim'));
vm.runInContext('renderOutputs(body)',ctx);
assert(allText(elements.get('digest-preview')).includes('Source-supported synthesis · not verbatim'));
elements.get('download').onclick();
assert.strictEqual(saved.payload.sections.ISSUE.status,'source_supported_synthesis');
assert.strictEqual(saved.payload.sections.ISSUE.text,'Whether payment is due.');
console.log('Delivery UI tests passed: fail closed, collapsed raw text, safe copy/export, explicit diagnostic export.');
