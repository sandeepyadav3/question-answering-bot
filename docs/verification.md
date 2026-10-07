# Verification

Completed for the final implementation checkpoint:

| Check | Result |
| --- | --- |
| Python suite | 172 passed; evaluation-runner suite 4 passed |
| Ruff lint | Passed |
| Frontend JavaScript syntax | Passed |
| Installed dependency compatibility | Passed |
| Python wheel build | Passed |
| Wheel contains HTML, CSS, JS, sample PDF and tokenizer vocabulary | Passed |
| Desktop browser layout, 1200px | Passed and visually reviewed |
| Mobile browser layout, 390px | Passed and visually reviewed; no horizontal overflow |
| Browser loading/error/retry states | Passed with mocked API responses |
| Answered/not-found/error cards and expandable evidence | Passed |
| Downloaded JSON matches API response | Passed |
| HTML-looking source content renders as text | Passed; no injected elements/scripts |
| Docker build, non-root runtime and container healthcheck | Passed |
| Live OpenAI sample JSON and PDF requests | Passed; answered evidence and unsupported `not found` checked |
| Grounding corpus against final container | 12/12 passed; all four reported metrics 100% |

Python verification used Python 3.12/Linux and the committed lockfile versions.
The test suite has one upstream Starlette warning about its HTTPX TestClient
compatibility; the tests pass. Browser verification used Playwright 1.51.1 and
Chromium 134. Browser responses were intercepted, so the checks made no paid calls.

## Re-run the browser checks

Start the application in one terminal. In a second terminal, from the project root:

```bash
npm install --no-save --package-lock=false playwright@1.51.1
npx playwright install chromium
node tests/ui-smoke.cjs
```

`BASE_URL` defaults to `http://127.0.0.1:8000`. Screenshots go to a temporary
directory printed by the script; `SCREENSHOT_DIR` can override it. The browser
script intercepts QA requests and does not send questions to OpenAI. Node/Playwright
are optional development tools; the application itself needs no Node runtime.

The grounding run used `gpt-4o-mini`, `text-embedding-3-small`, fetch-k 20,
minimum similarity 0.15, top-k 5 and MMR lambda 0.7. See
[grounding evaluation](grounding-evaluation.md) for case coverage and limitations.
