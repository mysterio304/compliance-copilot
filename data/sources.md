# Corpus sources

Five EAEU/Customs Union technical regulations, electronics scope, v1.

## Provenance chain

For each regulation: the authoritative citation (Commission/Council decision
number and date, taken from the regulation's own preamble text), the canonical
public-registry record, the mirror this text was actually retrieved from, and
why.

| Regulation | Authoritative act | Canonical registry (docs.cntd.ru) | Retrieved from |
|---|---|---|---|
| ТР ТС 004/2011 — О безопасности низковольтного оборудования | Решение Комиссии Таможенного союза от 16.08.2011 № 768 | [document/902299536](https://docs.cntd.ru/document/902299536) | meganorm.ru mirror |
| ТР ТС 020/2011 — Электромагнитная совместимость технических средств | Решение Комиссии Таможенного союза от 09.12.2011 № 879 | [document/902320551](https://docs.cntd.ru/document/902320551) | meganorm.ru mirror |
| ТР ЕАЭС 037/2016 — Об ограничении применения опасных веществ в изделиях электротехники и радиоэлектроники | Решение Совета ЕЭК от 18.10.2016 № 113 | [document/420387089](https://docs.cntd.ru/document/420387089) | meganorm.ru mirror |
| ТР ТС 010/2011 — О безопасности машин и оборудования | Решение Комиссии Таможенного союза от 18.10.2011 № 823 | [document/902307904](https://docs.cntd.ru/document/902307904) | meganorm.ru mirror |
| ТР ТС 008/2011 — О безопасности игрушек | Решение Комиссии Таможенного союза от 23.09.2011 (с изм. на 17.03.2017) | [document/902303210](https://docs.cntd.ru/document/902303210) | meganorm.ru mirror |

Retrieved: 2026-08-03.

## Why these PDFs were locally rendered rather than downloaded as-is

The primary PDF hosts for this corpus — docs.cntd.ru, eec.eaeunion.org (the
Eurasian Economic Commission's own site), gost.ru, and consultant.ru — were
unreachable from this environment: some return a Cloudflare bot-challenge page
instead of the file, others time out on every request. This looked at first
like it might be a deliberate network restriction, but a wider connectivity
test showed it isn't a blanket block (github.com, huggingface.co, and even
some other .ru domains loaded fine) — it's specific-site flakiness/anti-bot
protection on those particular hosts.

meganorm.ru mirrors the same regulation texts as clean, uncompressed HTML
(itself a Word-to-HTML export of the original government document, based on
the `MsoNormal`/`h1` CSS classes present in the markup) and was reachable,
though it needed retries under load. **The text content pulled from there is
the real, verbatim regulation text** — matching the authentic decision
numbers, dates, and article structure — not paraphrased or synthesized.

Since the ingestion pipeline (`ingest/pdf_parser.py`) is built around parsing
PDFs with Docling, and the only accessible copies of this text were HTML, each
document was rendered into a real PDF locally: the HTML was parsed into
ordered heading/paragraph blocks (`scripts/` one-off, not part of the shipped
package), and rendered via ReportLab into an actual PDF file, preserving
headings as real bold/larger-font structural elements so Docling's layout
model has the same kind of visual signal it would see in an official PDF
export of the same Word document. This was verified, not assumed: Docling
correctly recovers `Статья N` / `Раздел N` / `Приложение N` as
`section_header` items from these rendered PDFs (see `tests/test_ingest.py`).

**What this means for corpus fidelity:** the regulatory *text* is authentic
and unmodified (aside from HTML whitespace normalization). The PDF *file
bytes* are not the government's original PDF — they're a faithful local
re-rendering of the same text. If direct network access to docs.cntd.ru or
eec.eaeunion.org becomes available later, the source PDFs in
`data/raw_pdfs/` should be replaced with the originals and re-ingested; the
manifest (`data/corpus_manifest.yaml`) and ingestion pipeline don't need to
change either way.

## A structural assumption that turned out to be wrong

The original brief assumed all five regulations share one heading structure
(статьи/пункты/приложения). That holds for four of them (ТР ТС 004, 008, 010,
020 — all 2011 Customs Union regulations, all using `Статья N.` headings).
**ТР ЕАЭС 037/2016 does not**: as a later Eurasian Economic Union (not Customs
Union) regulation, it structures its body as Roman-numeral `Раздел`s (`I.
Область применения`, `II. Основные понятия`, ...) instead of `Статья N.`, and
its appendices are consistently `Приложение № N` rather than the unnumbered or
inconsistently-numbered `Приложение` seen elsewhere. `ingest/clause_extractor.py`
handles both patterns explicitly (`heading_to_clause_path`) rather than
assuming one shape.

## Reproducing the corpus

```bash
uv run python scripts/ingest_corpus.py   # data/raw_pdfs/*.pdf -> data/processed/*.json
uv run python scripts/corpus_stats.py    # measured document/clause/chunk counts
```
