# AI usage

Course rule (DS607 guidelines): GenAI is allowed if we remain able to justify every modelling
choice, and every use is documented here.

Add one row every time you use an AI tool, before your deadline. The last column is required:
say how you checked the output (re-ran it, compared with a hand calculation, read the docs, etc.).

| Date | Member | Tool | File | What it was used for | How the output was checked |
| --- | --- | --- | --- | --- | --- |
| 2026-10-21 | Zhang Enhao | Claude | proposal PDF | Drafting the one-page proposal from the group's design | Read and approved by all members |
| 2026-10-21 | Zhang Enhao | Claude Code | repository skeleton | Folders, module stubs, notebook headers, README and templates | Reviewed every file before pushing |
| 2026-10-22 | Deng Shenbojia | Claude | src/data.py, src/strategy.py | Drafting fetch_klines, load_daily and build_series (A1) | Offline test with a mocked Binance API (paging, fallback URL, missing-day assert); SMA lag recomputed by hand for one day; checked d = -(1 - s) r and the cost total; real-data numbers compared with the trial run after the local pull |
