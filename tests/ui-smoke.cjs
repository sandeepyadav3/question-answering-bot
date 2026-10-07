// Optional browser check: npm install --no-save --package-lock=false playwright@1.51.1
// Then: npx playwright install chromium && node tests/ui-smoke.cjs
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const os = require('node:os');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const screenshots = process.env.SCREENSHOT_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'document-qa-ui-'));
  fs.mkdirSync(screenshots, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH});
  try {
    const page = await browser.newPage({viewport: {width: 1200, height: 950}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:8000');
    assert.equal(await page.title(), 'Document QA');
    await page.screenshot({path: path.join(screenshots, 'desktop.png'), fullPage: true});
    await page.locator('#questions-file').setInputFiles(path.join(__dirname, '../app/static/examples/questions.json'));
    await page.locator('#document-file').setInputFiles(path.join(__dirname, '../app/static/examples/document.json'));
    // Every provider request is intercepted: browser checks never make paid API calls.
    await page.route('**/api/v1/qa', route => route.fulfill({status: 503, contentType: 'application/json', body: JSON.stringify({error: {code: 'PROVIDER_NOT_CONFIGURED', message: 'Missing configuration', request_id: 'test-request'}})}));
    await page.locator('#submit-button').click();
    await page.waitForFunction(() => document.querySelector('#notice').dataset.kind === 'error');
    assert.match(await page.locator('#notice').innerText(), /not configured/);
    assert.equal(await page.locator('#submit-button').isEnabled(), true);
    await page.unroute('**/api/v1/qa');
    const injection = '<img src=x onerror="window.xss=true">';
    const response = {request_id: 'sample-request', results: [
      {question: 'Which cloud provider is used?', status: 'answered', answer: 'The service uses AWS.', citations: [{source_type: 'pdf', page: 1, snippet: 'Production uses AWS.'}]},
      {question: 'Is there an ISO 27001 certification?', status: 'not_found', answer: 'not found', citations: []},
      {question: injection, status: 'error', answer: null, citations: [], error: {code: 'LLM_TIMEOUT', message: 'This question timed out. Please try again.'}}
    ], metadata: {questions_answered: 1, questions_not_found: 1, questions_failed: 1, latency_ms: 1400}};
    await page.route('**/api/v1/qa', async route => {
      await new Promise(resolve => setTimeout(resolve, 200));
      await route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify(response)});
    });
    await page.locator('#submit-button').click();
    assert.equal(await page.locator('#submit-button').isDisabled(), true);
    await page.locator('#results-section').waitFor({state: 'visible'});
    assert.equal(await page.locator('.result-card').count(), 3);
    assert.match(await page.locator('#results-summary').innerText(), /1 answered · 1 not found · 1 failed/);
    assert.equal(await page.locator('#results-list img').count(), 0);
    assert.equal(await page.evaluate(() => window.xss), undefined);
    assert.match(await page.locator('.result-card').nth(2).innerText(), /<img src=x/);
    await page.locator('summary').click();
    assert.equal(await page.locator('blockquote').innerText(), 'Production uses AWS.');
    await page.screenshot({path: path.join(screenshots, 'results.png'), fullPage: true});
    const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-button').click()]);
    assert.equal(download.suggestedFilename(), 'document-qa-results.json');
    assert.deepEqual(JSON.parse(fs.readFileSync(await download.path(), 'utf8')), response);
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(screenshots, 'mobile.png'), fullPage: true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    assert.deepEqual(errors, []);
    console.log('Browser checks passed: desktop/mobile, errors, loading, result states, evidence, JSON download, safe text rendering.');
    console.log('Screenshots: ' + screenshots);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
