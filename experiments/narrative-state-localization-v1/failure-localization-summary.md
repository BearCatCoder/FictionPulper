# Failure Localization

**Final classification: D SYNTHETIC_SHORTCUT.** Weak representation is a contributing limitation, but narrow one-family evidence does not justify category B. The decisive mismatch is 97.65% original synthetic pair ranking versus failed balanced reversal and poor original-context 13-fact scoring.

## Final Questions

1. **Is narrative state linearly available in frozen hidden states?** Only weakly and in isolated families; no model/family is robust on both test and generalization. Contrastive object_location is the sole convincing relative improvement, not broad representation.
2. **Does LM next-token scoring use earlier narrative state under a gold/correct context?** Not reliably. Contrastive reaches only 4/13 top-1 on historical gold-prefix forced choice and only 0.5982 direction accuracy on balanced swaps.
3. **Does Contrastive-v1's 97% ranking accuracy survive counterfactual context reversal?** No. Contrastive reverses 11/56 (0.1964), below the 0.25 random-pair benchmark; removed and irrelevant controls retain large candidate margins.
4. **At what rollout length does state-consistency scoring begin to collapse?** No collapse length is identifiable because the gold baseline is already absent: Contrastive starts at 4/13 with mean margin -0.2288, then fluctuates non-monotonically.
5. **Does correcting an erroneous rollout restore the state-consistency margin?** Yes. Replacing every generated suffix with the exact gold suffix restores every candidate score exactly (maximum absolute recovery error 0.0), but it restores the same weak gold-context baseline rather than correct state use.
6. **What is the primary failure localization?** D SYNTHETIC_SHORTCUT, with weak representation as a contributing limitation: 97.65% original synthetic pair ranking does not transfer to balanced context reversal or 13-fact gold scoring.

## Three Levels

| Level | Compute5 | Narrative | Contrastive | 50M |
|---|---|---|---|---|
| representation | Weak/isolated | Weak/isolated | Weak/isolated; object_location relative improvement only | Not probed |
| gold-prefix forced choice | 4/13 | 3/13 | 4/13 | 7/13 |
| historical free generation | 0/13 | 0/13 | 0/13 | 0/13 |

No `/13` probe score is fabricated because the probe task and historical facts differ.

## Controls and Limitations

- Controlled synthetic swaps use held-out vocabulary but templated prompts.
- Probe labels are lexical-rank labels, not the historical 13 facts.
- Each model is represented by one selected checkpoint seed.
- The historical suite contains only 13 facts.
- Exact-phrase contradiction checking is conservative.
