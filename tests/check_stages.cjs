// Browser contract test using synthetic API data; does not claim model accuracy.
const {chromium} = require('C:/Users/drylr/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage();
    const errors = []; page.on('pageerror', e => errors.push(e.message));
    await page.route('**/api/status', route => route.fulfill({json:{classifier_ready:true,ml_dependencies_ready:true}}));
    const fixture = {case_id:'TEST CASE', blocks:{FACT:'Original facts.',ISSUE:'Is the contract binding?',RULING:'Original ruling.'},
      sentences:[{id:1,text:'Source fact.',role:'FACT',confidence:0.9},{id:2,text:'Other content.',role:'OTHER',confidence:0.8}],
      chunks:{FACT:1,ISSUE:1,RULING:1},warnings:[],exemplar_case_ids:[]};
    await page.route('**/api/summarize', route => route.fulfill({json:fixture}));
    await page.goto('http://127.0.0.1:8000');
    await page.locator('#case').fill('TEST CASE');
    assert.equal(await page.locator('#text').isVisible(), false);
    await page.locator('#input-method').selectOption('text');
    assert.equal(await page.locator('#text').isVisible(), true);
    assert.equal(await page.locator('#file').isDisabled(), true);
    await page.locator('#input-method').selectOption('pdf');
    assert.equal(await page.locator('#file').getAttribute('required'), '');
    await page.locator('#file').setInputFiles({name:'decision.pdf',mimeType:'application/pdf',buffer:Buffer.from('%PDF-1.4 synthetic browser fixture')});
    await page.locator('#generate').click();
    await page.locator('#bert-output table').waitFor();
    assert.equal(await page.locator('#bert-output tbody tr').count(), 2);
    assert.match(await page.locator('#bert-output').innerText(), /OTHER/);
    await page.locator('[data-panel="pegasus-panel"]').click();
    assert.equal(await page.locator('#pegasus-panel').isVisible(), true);
    assert.match(await page.locator('#pegasus-output').innerText(), /Original facts/);
    await page.locator('[data-panel="final-panel"]').click();
    await page.locator('#block-FACT').fill('Edited facts.');
    assert.match(await page.locator('#digest-preview').innerText(), /Edited facts/);
    assert.match(await page.locator('#pegasus-output').innerText(), /Original facts/);
    const downloadEvent = page.waitForEvent('download');
    await page.locator('#download').click();
    const download = await downloadEvent;
    const data = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert.equal(data.blocks.FACT, 'Edited facts.');
    assert.equal(data.original_blocks.FACT, 'Original facts.');
    await page.route('**/api/summarize', route => route.fulfill({status:503,json:{detail:'Generator setup failed: missing dependency.'}}));
    await page.locator('#generate').click();
    await page.locator('#notice').filter({hasText:'Generator setup failed'}).waitFor();
    assert.match(await page.locator('#bert-output').innerText(), /Generator setup failed: missing dependency/);
    assert.equal(await page.locator('#download-bert').isDisabled(), true);
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.deepEqual(errors, []);
    console.log('PASS: stage navigation, OTHER visibility, immutable model output, live final preview, JSON export, specific failure display, mobile width.');
  } finally { await browser.close(); }
})().catch(e => {console.error(e);process.exitCode=1;});
