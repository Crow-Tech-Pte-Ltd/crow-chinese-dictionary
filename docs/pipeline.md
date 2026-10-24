# Pipeline and reproducibility guide

Run the commands below from the repository root, not from `docs/`.
For the current public release, see the [project overview](../README.md) and
[release manifest](../data/release/manifest.json). Examples using a local Hanzi Desk
checkout require its separately retained source files; private work directories
and API credentials are intentionally absent from this public repository.

A source-attested Chinese dictionary project for Crow Tech. The pipeline finds
vocabulary and definition-language gaps, creates reviewable drafts, and exports
only entries that pass explicit publication gates. It does not infer that a word
is missing merely because it failed one search.

Python 3.10+; standard library only. No installation, paid API or model download
is needed for inventory, queue preparation, dry runs or tests.

## What is a gap?

The reference universe is the existing Hanzi Desk catalogue, its retained licensed
originals, the retained HSK source, and explicitly registered upstream snapshots.
A candidate is compared by spoken variety,
source-backed script aliases and lexical reading. Mandarin and Cantonese stay
separate. Traditional aliases are matched conditionally on reading; there is no
blind script-alias union. For example, `妳` must never merge `你 ni3` with `奶 nai3`.

Separate queues distinguish:

- A word or phrase absent from every current target-language pack.
- A covered word with a possible additional reading or meaning, requiring review.
- An optional pack already containing the definition, needing no new paid work.
- Existing machine-only coverage, which is a quality-upgrade question rather than
  a missing-word claim.
- Missing, ambiguous, corrupt or unsupported pronunciation evidence.
- Raw-source leads wholly absent from the current application.

This cannot enumerate the exact difference against Pleco's proprietary
collection without a permitted comparison list. It measures gaps against the
identified reference materials, not every Chinese word in existence.

English Wiktionary's original website import is filtered to CC-CEDICT headwords.
Rescanning that filtered file cannot discover the English entries it omitted.
The optional full upstream snapshot below removes that discovery restriction;
broader retained localized originals supply additional source records.

## Fetch and distill the full English source

The reviewed upstream source is Kaikki's complete **Chinese bucket from English
Wiktionary**, extracted on 21 October 2026 from the 1 October 2026 dump. The
compressed original is 155,995,894 bytes. Public URL, checksum, dates, copyright
links and download metadata are in `reports/upstream-source-provenance.json`.

Verify the retained source without any network request:

```sh
python3 -m crowdict.fetch_source --root work/upstream --verify-only
```

On a fresh checkout, explicitly download the checksum-pinned official snapshot:

```sh
python3 -m crowdict.fetch_source --root work/upstream --download
```

The downloader allows only official Kaikki HTTPS downloads, enforces a 500 MiB
cap, verifies size and SHA-256 before promoting the file, and skips an already
valid download. Kaikki's URLs roll forward: if their contents change, this command
fails rather than silently accepting a different snapshot. Review the new source
and update the provenance deliberately, or retain the reviewed original locally.

Create the smaller reusable English source and its external catalogue:

```sh
python3 -m crowdict.distill \
  --input work/upstream/public/data/kaikki.org-dictionary-Chinese.jsonl.gz \
  --root work/upstream --snapshot-date 2026-10-21 --dump-date 2026-10-01
```

The importer streams JSONL records instead of materialising the 1.2 GB expanded
source. It requires explicit standard Mandarin Pinyin; generic Chinese/Han labels
alone are insufficient. Other Chinese varieties, composite variety tags such as
Taiwanese-Hokkien, uncertain regional/dialectal senses and extraction-error senses
are excluded. Multiple source readings remain unresolved. Exact source spellings,
sense IDs, scope tags and raw pronunciation fields remain attached; script aliases
and pronunciations are never invented. Quotations, examples and media are omitted.

The separate Mandarin and Cantonese download buckets primarily index romanizations
and characters. They are research references, not substitutes for the complete
Chinese-word English source and not inputs to this Mandarin distillation.
Kaikki marks these postprocessed language downloads deprecated; its newer complete
raw all-language gzip is approximately 2.9 GB and was not downloaded under this cap.

`reports/upstream-source-qa.json` and `reports/upstream-distill-metrics.json` contain
public aggregate checks only. Imported English text remains attributed source
material under CC BY-SA 4.0, not original Crow or human-reviewed definitions.

## Build and inspect the local inventory

Run from this directory, beside the existing website checkout:

```sh
python3 -m unittest discover -s tests -v
python3 -m crowdict.inventory build --web-root ../hanzi-desk-web \
  --extra-catalog work/upstream/extra-catalog.json
python3 -m crowdict.queues --db work/inventory.sqlite --out work/queues
python3 -m crowdict.inventory jobs --web-root ../hanzi-desk-web --language en --limit 10000
python3 -m crowdict.plan --web-root ../hanzi-desk-web --languages all --per-language-limit 1000
```

`work/inventory.sqlite` is a derived local index; it is ignored by Git. The public
summary is `reports/inventory-summary.json` and the definition-language comparison
is `reports/language-coverage.csv`. `work/jobs/jobs-*.jsonl.gz` contains the
prioritised, source-backed definition tasks. `held-*` and `reuse-*` remain separate.
There is no full 32-language Cartesian-product job dump wasting disk space.

Every job has a deterministic ID, canonical word, source spellings, fixed lexical
reading, spoken variety, target definition language, priority, evidence hashes,
source references, licences and hold reasons. Named entities, low-frequency
candidates, suspicious variants and existing-target aliases/readings are held
from the automatic generation queue. HSK and word-frequency data prioritise
useful vocabulary; they do not prove a candidate's dictionary identity is correct.

Explicit pinyin can be recovered from eligible raw Wiktionary fields. Only one
unambiguous, valid source reading can fill a blank; a model never supplies it.

The external catalogue registers source material, not existing app coverage:
its sources stay `baseline: false`. An English definition found there goes to
exact source reuse when its reading is resolved; ambiguous target-English evidence
goes to review rather than paying a model to recreate an already available meaning.
The distilled source catalogue uses a portable relative `root`; keep it beside
its registered source files when moving the project. Omit `--extra-catalog` only when intentionally auditing the website
and its retained sources without the fresh upstream snapshot.

## Generate: dry-run first

Current provider research is in `reports/provider-research.json` and
`config/providers.example.json`. Recheck models, pricing and terms before running
at a later date. Token-plan credits are not assumed to be dollars. At the research
date MiniMax documents credit-pack Credits as 1,000 = $1, separate from included
subscription allowance. The plan usage API reports percentage windows.

```sh
python3 -m crowdict.generate \
  --jobs work/jobs/jobs-en.jsonl.gz --ledger work/pilot-ledger.sqlite \
  --provider minimax --model MiniMax-M3.1-Flash \
  --input-usd-per-million .15 --output-usd-per-million .60 \
  --max-usd .50 --max-calls 50 --max-total-tokens 1000000 \
  --max-jobs 20 --max-output-tokens 2048 --billing token-plan
```

The command above reads no key and makes no request. Actual inference requires
both `--execute` and `--key-file /path/to/your/.minimax-key`. With MiniMax token-plan
billing it also requires `--allow-credit-pack-debit`, acknowledging that purchased
packs may be consumed after included quota. The file must contain the appropriate
Subscription Key, not a standard pay-as-you-go key. Never paste a key into logs,
source files, a prompt, the repository or a command-line argument value.

MiniMax requests use low reasoning effort and capped completion tokens, including
thinking. The direct endpoint does not promise native JSON schema; local checks
are mandatory. OpenRouter requests require supported structured output, deny
provider data collection, pin upper prices and disable provider fallback.

The current generator protocol is `crow-definition-v3`. Source POS context and
raw source senses, including normalized tags and free-text `raw_tags`, enter the
actual prompt. Explicit usage restrictions such as archaic, slang or regional
must remain in the target-language definition for the corresponding sense.
Generated `pos` and `register` fields remain empty in this phase; a model must not
invent labels such as “neutral” or “noun phrase”. Foreign-language source evidence
is expected and is translated for the requested definition language.

A persistent SQLite ledger reserves conservative input/output cost before each
request. Counts, cost holds, retries, timeouts and interruptions survive restarts
and coordinate across workers. Unknown charges retain the reservation. Successful
outputs are skipped on resume. Do not delete the ledger to continue past a cap.
`--raise-budget-caps` explicitly permits increasing limits while preserving all
past usage. Subscription USD figures are conservative PAYG-equivalent reservations,
not a claim of actual dollars charged. A provider price/usage anomaly halts work.

For a definitive authentication/payment rejection after correcting the key:

```sh
python3 -m crowdict.ledger --ledger work/pilot-ledger.sqlite --summary
python3 -m crowdict.ledger --ledger work/pilot-ledger.sqlite \
  --requeue-rejected --job-id JOB_ID
```

This operation preserves all counters and retry limits. Unknown outcomes,
quarantined requests and successful drafts cannot be requeued this way.

## Independent checking and publication

Generator results remain `machine_draft`. JSON structure, references and model
confidence do not prove a correct dictionary meaning. Source extracts themselves
can contain errors, and several language packs can repeat the same upstream data.

A prompt change creates a new generation fingerprint; it does not erase the old
attempts or their cost. First materialize one active draft per job from the exact
current prompt version and model:

```sh
python3 -m crowdict.snapshot \
  --ledger work/pilot-ledger.sqlite --jobs work/jobs/jobs-en.jsonl.gz \
  --prompt-version crow-definition-v3 --model MiniMax-M3.1-Flash \
  --out work/active-drafts-v3.jsonl.gz
```

The snapshot requires the stored generation-time evidence hash to match the
current job, rechecks identity and structure, and selects the latest matching
valid draft per job. Its report records chosen and superseded fingerprints and
held stale versions. It never falls back to an older prompt version. Use this
snapshot, rather than every historical row in the ledger, for review, audition
and publication.

The current reviewer protocol is `crow-semantic-review-v5`. It checks each sense,
Chinese identity, attested reading, target-language meaning, preservation of
explicit source usage restrictions, and Chinese lexical/numeric source
compatibility. Related source families do not count as independent corroboration. A French or German source gloss is legitimate
evidence for an English definition; the target-language check applies to the
draft. The checker uses a different model from the generator.

Use **GLM 5.3 as the main checker**, with **DeepSeek V4 Pro for a small additional
sample**, retaining their separate decision files. This is a scheduling choice,
not a claim that either model guarantees dictionary correctness. Example dry run
for the main checker:

```sh
python3 -m crowdict.review \
  --jobs work/jobs/jobs-en.jsonl.gz --results work/active-drafts-v3.jsonl.gz \
  --ledger work/pilot-ledger.sqlite --provider openrouter \
  --model z-ai/glm-5.3 \
  --input-usd-per-million 1.10 --output-usd-per-million 2.20 \
  --max-usd .50 --max-calls 50 --max-total-tokens 1000000 --max-jobs 20 \
  --workers 2 --out-decisions work/glm-decisions-v5.jsonl \
  --out-checks work/glm-checks-v5.jsonl
```

For a second-model smoke sample, the command below checks the first five eligible
snapshot records. That is an ordered smoke sample, not a random accuracy estimate;
use a deliberately selected challenging job/snapshot subset for broader QA:

```sh
python3 -m crowdict.review \
  --jobs work/jobs/jobs-en.jsonl.gz --results work/active-drafts-v3.jsonl.gz \
  --ledger work/pilot-ledger.sqlite --provider openrouter \
  --model deepseek/deepseek-v4-pro \
  --input-usd-per-million 1.10 --output-usd-per-million 2.20 \
  --max-usd .50 --max-calls 50 --max-total-tokens 1000000 --max-jobs 5 \
  --workers 1 --out-decisions work/deepseek-sample-decisions-v5.jsonl \
  --out-checks work/deepseek-sample-checks-v5.jsonl
```

Actual review additionally needs `--execute --key-file /path/to/.openrouter-key`.
Generation and checking share the cumulative budget, including older prompt
versions and both checkers. Review supports one to four bounded workers and stops
new dispatch after authentication, payment, quota or unknown-outcome failures;
already in-flight requests may still complete. The example limits are pilot
ceilings. Raise them explicitly with `--raise-budget-caps` only within the total
authorised budget; previous calls and reservations still count.
Same-model checks cannot approve an entry. Model checks are labelled
`machine_checked`, never `human_reviewed`.
Decisions are bound to exact generation-time evidence and draft hashes; stale
approvals cannot apply to modified definitions.

```sh
python3 -m crowdict.validate --jobs work/jobs/jobs-en.jsonl.gz \
  --results work/active-drafts-v3.jsonl.gz --out data/english-preview
```

Default export accepts only genuine human-approved entries. Add `--decisions`
with explicit review decisions and `--include-machine-checked` to include actual
independent-model approvals. Exact same-language source imports have their own
explicit `--include-source-imports` gate. An empty accepted export is an honest
outcome while reviews or source evidence are missing.

Combine only the decisions for this active snapshot when using both checkers:

```sh
cat work/glm-decisions-v5.jsonl work/deepseek-sample-decisions-v5.jsonl \
  > work/combined-decisions-v5.jsonl
python3 -m crowdict.validate --jobs work/jobs/jobs-en.jsonl.gz \
  --results work/active-drafts-v3.jsonl.gz \
  --decisions work/combined-decisions-v5.jsonl --include-machine-checked \
  --out data/english-preview
```

A rejection or uncertainty from the additional DeepSeek sample remains a hold;
do not replace an unfavourable check with a more agreeable verdict. Draft/evidence
hash checks prevent stale approvals from attaching to a modified definition.

The exporter writes deterministic gzip JSONL, checksummed manifests, actual-source
notices and website-shaped arrays. Existing-site integration is a separate release;
these files do not modify the application. Full-corpus static delivery may require
bounded shards to respect hosting limits; do not upload oversized packs or
thousands of new per-word files. Audio is referenced by canonical word, reading
and spoken variety, so languages reuse the same Chinese clip. Audio existence and
correctness are not inferred from definition success; `audioStatus` starts as
`not_checked`. No TTS generation is part of this project.

## Local audition page

```sh
python3 -m crowdict.audition --jobs work/jobs/jobs-en.jsonl.gz \
  --results work/active-drafts-v3.jsonl.gz --out work/review
python3 -m http.server 8768 --bind 127.0.0.1 --directory work/review
```

Open http://127.0.0.1:8768/. The page has no external assets or analytics and
receives no API key. It displays source evidence and drafts, and can download
explicit human review decisions. With no successful drafts it displays candidates
without inventing definitions. Serve only the generated review directory, never
the project root or a directory containing credentials.

## Resumable bounded batches

`crowdict.batch` runs generation, an active snapshot, GLM 5.3 checking and gated
export in bounded batches. It uses the same cumulative ledger for both providers.
The default is a local dry run with no key reads, API requests or output export:

```sh
python3 -m crowdict.batch \
  --jobs work/jobs/jobs-en.jsonl.gz --ledger work/pilot-ledger.sqlite \
  --max-usd 2 --max-calls 1000 --max-total-tokens 10000000 \
  --batch-size 64 --max-jobs 256 --workers 4 \
  --out work/batches/english --export data/english-machine-checked
```

Actual work requires `--execute` and both explicit `--minimax-key-file` and
`--openrouter-key-file` paths. The example is a $2 conservative shared ceiling,
not permission to spend every available account balance. Existing ledger usage
still counts. If these caps exceed the ledger's earlier pilot limits, execution
refuses unless the operator deliberately adds `--raise-budget-caps` within the
authorised total; no stage silently raises a cap or starts a fresh budget.

Keep `work/batches/english/jobs.jsonl.gz` and the same work directory to resume the
fixed job scope. Per-batch logs, snapshots and decisions remain in numbered
subdirectories. Aggregate drafts/decisions and the atomic `status.json` live in
`work/batches/english/`; check that status and the shared ledger before resuming.
Provider/quota or budget stop markers prevent scheduling another batch. Completed
fingerprints are skipped, unknown outcomes remain held, and machine-checked
exports remain explicitly labelled automatic checks rather than human review.
Run an additional bounded DeepSeek sample as described above before accepting a
broader release; batch processing itself does not publish or integrate the data.

## Current release and continuation

The completed multilingual run is bundled in `data/release/`: **46,791 entries**
covering **41,446 distinct canonical headwords** across **32 definition languages**.
It combines **40,682 attributed source imports** and **6,109 independently
model-checked synthetic definitions**; none is labelled human-reviewed.
English includes 40,682 source imports and 288 synthetic entries. The release
manifest records checksums, source snapshots and the 2,368 pending jobs excluded
from this export scope. Counts are a snapshot; the release manifest is authoritative.

`reports/corpus-discovery-current.json` separately counts source imports with no
spelling/alias in any baseline pack: 23,388 distinct canonical heads. This is not
an exact Pleco comparison or proof of full dictionary coverage. The final English
candidate scope also holds 31,563 entries for reading, source or lexical review.
The full supported-language reference and limitations are in
`reports/inventory-summary.json`, with detailed private queues under
`work/jobs-current/` and `work/queues-current/`.

Prepare a fresh priority scope, copying same-language evidence before paid work:

```sh
python3 -m crowdict.plan --web-root ../hanzi-desk-web --languages all \
  --english-limit 100000 --per-language-limit 1000 --out work/jobs-current
python3 -m crowdict.reuse --jobs work/jobs-current/reuse-en.jsonl.gz \
  --out work/english-source-current.jsonl.gz
```

The completed run was prepared from a scope of 8,464 eligible generation jobs,
including 328 English jobs already attempted at the time of preparation. The other-language limit is the first 1,000 candidates
per language, before eligibility holds. `jobs-balanced.jsonl.gz` keeps English
first and then interleaves the other languages by priority. It is not a claim that
all vocabulary in 32 languages has been completed.

To reproduce or resume a bounded multilingual run, retain the same cumulative ledger:

```sh
python3 -m crowdict.batch \
  --jobs work/jobs-current/jobs-balanced.jsonl.gz \
  --ledger work/pilot-ledger.sqlite \
  --out work/multilingual-batch --export work/multilingual-export \
  --max-usd 50 --openrouter-max-usd 20 \
  --max-calls 20000 --max-total-tokens 400000000 \
  --batch-size 64 --workers 2 --raise-budget-caps \
  --additional-decisions work/deepseek-v5-pilot-decisions.jsonl
```

This example is a dry run. Actual work additionally requires `--execute` and the
two explicit key-file paths. The $50 combined ceiling is a conservative reservation
in USD-equivalent units, including MiniMax subscription usage; it is **not a claim
of $50 cash spending**. OpenRouter has its own persistent **$20** ceiling, including
previous known charges and unknown-outcome reservations. Provider caps cannot be
removed by omitting a later option. Standalone generation/review CLIs also accept
`--provider-max-usd`. Raising any stored ceiling requires the explicit raise flag.
No top-up, paid GPU, PAYG-key substitution or production publication is performed.

Before generating each batch, the runner snapshots valid completed drafts and
excludes them from new generation, even when harmless display metadata changed.
Use the same fixed work directory to resume. Different scopes require a new work
directory while retaining the ledger. To drain and stop between batches, create
`work/multilingual-batch/STOP`; remove that file deliberately before resuming.
Do not change prompts or input scope while a batch is running.

Bundle current source imports and checked generation into the public release:

```sh
python3 -m crowdict.bundle \
  --jobs work/jobs-current/reuse-en.jsonl.gz \
  --jobs work/jobs-current/jobs-balanced.jsonl.gz \
  --results work/english-source-current.jsonl.gz \
  --results work/multilingual-batch/drafts.jsonl.gz \
  --decisions work/multilingual-batch/decisions.jsonl.gz \
  --include-source-imports --include-machine-checked --out data/release
```

Bundling is offline and revalidates every record. Run it after generation finishes
or reaches a provider/budget stop, using only the matching final scope and decisions. Partial batches,
uncertain readings and rejected definitions remain explicit in the queues. Import
`data/release/website/import-manifest.json` as an optional dictionary source in a
separate website/native release, preserving per-entry origin labels, script forms,
source grammar and personal data identities. Full audit metadata stays in
`entries.jsonl.gz`; the website-shaped packs are compact adapters. Audio is matched
by Chinese identity/reading/variety and remains `not_checked` in this dataset.

## Licensing and release boundaries

Code is MIT. Compatible derived dictionary data is CC BY-SA 4.0 with retained
upstream attribution and notices; raw sources retain their original licences.
See [data licences](../DATA_LICENSE.md). ND glosses are excluded from prompts and public derivatives;
WordNet component licences need review. Machine-translated sources are marked and
are not independent proof. This is not a copy of Oxford, ABC or New Century.

Keys, API responses, account/quota data and work databases are private and ignored.
Only code, tests, public aggregate reports, licence policies and accepted dataset
exports belong in this public repository. Dataset export does not itself modify
or deploy Hanzi Desk. Website and native integration preserve source credits,
origin labels, script forms and existing personal-data identities.
