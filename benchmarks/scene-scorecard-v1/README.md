# Scene Scorecard v1

This directory freezes the product-facing scene protocol owned by issue #27. It
is separate from and does not modify Narrative Benchmark v2.

The suite contains five previously exposed development/regression premises and
five newly authored post-selection premises. `prompts.jsonl` records exposure,
genre, premise, motive, and four atomic opening facts for every prompt. The five
`scene-post-*` prompts are reserved from issue #8 training, validation, test
authoring, candidate selection, and tuning. They may be opened only for the
frozen evaluation represented here.

Run the sealed 50M baseline and export its deterministic blinded packet with:

```bash
python -m src.scene_scorecard baseline
```

The default write-once destination is
`runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1`. See
[`docs/scene-scorecard-baseline.md`](../../docs/scene-scorecard-baseline.md) for
review and reporting status. Automated word, lexical fact-mention, and
repetition fields are diagnostics only; promotion uses genuine independent
human review under `human-review-rubric.json`.

After two genuine independent reviewers complete separate copies of the packet,
combine their rows without changing packet fields. Add a complete
`reviewer_role: "adjudicator"` row from a third independent person whenever any
field differs or is `uncertain`, then run:

```bash
python -m src.scene_scorecard score-reviews \
  --manifest runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1/manifest.json \
  --packet runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1/blinded-review-packet.jsonl \
  --key runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1/private-review-key.json \
  --reviews runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1/completed-reviews.jsonl \
  --output runs/scene-scorecard-v1/fictionpulper-50m-data30m-v1/human-score-report.json
```

The scorer validates packet content and reviewer identities but cannot prove
real-world independence. That remains a documented human-process requirement.
The resulting human report retains every output score and includes a pooled
40-output summary, separate development/regression and post-selection summaries,
and separate greedy plus per-sampled-seed trial summaries. Stratified results are
descriptive; only the pooled complete suite drives the frozen promotion gates.
