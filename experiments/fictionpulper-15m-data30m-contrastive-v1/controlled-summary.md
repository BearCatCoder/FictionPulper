# Controlled Summary

**Classification: `OBJECTIVE_LEARNED_WITHOUT_NARRATIVE_TRANSFER`**

Contrastive-v1 learned to rank synthetic positive continuations above templated negatives, but this did not transfer to explicit fact retention or broad narrative-state representation.

## Controls

All three 15M models use 15,047,040 parameters, tokenizer-v1, 1,024-token context, seed 1337, step 2220 selection, and 135,886,355 positive LM target presentations. Narrative-v1 and Contrastive-v1 use the same 85/15 real/curriculum schedule; Contrastive-v1 adds lambda=0.25 ranking loss and about 39.55% extra forward-token compute by the recorded proxy.

## Exact 13-Fact Audit

`F/F` means greedy false and sampled false. The JSON scoring record contains every mode-specific evidence snippet.

| Fact ID | Exact fact | Compute5 | Narrative-v1 | Contrastive-v1 | Reason |
|---|---:|---:|---:|---:|---:|
| name_object_location_goal.character_name | Martin Vale | F/F | F/F | F/F | No continuation names Martin Vale or identifies its active character as him. |
| name_object_location_goal.object_possession | a brass key in his left coat pocket | F/F | F/F | F/F | Key mentions belong to other entities or locations; none preserves Martin's brass-key possession or left-pocket state. |
| name_object_location_goal.location | the Blackwater Hotel | F/F | F/F | F/F | No output identifies the current place as the Blackwater Hotel. |
| name_object_location_goal.stated_goal | reach room 317 before midnight | F/F | F/F | F/F | Generic room language does not preserve room 317, the midnight deadline, or Martin's goal. |
| relationship_secret.character_name | Jonah Reed | F/F | F/F | F/F | No continuation names Jonah Reed or maintains him as the active character. |
| relationship_secret.relationship | Elise Ward is Jonah's sister-in-law | F/F | F/F | F/F | Marriage and family words refer to unrelated generated characters; Elise, Jonah, and their sister-in-law relation are absent. |
| relationship_secret.secret | Jonah saw Elise's brother Daniel leave alive on the northbound train | F/F | F/F | F/F | Generic secrecy or seeing language does not preserve Daniel's survival, Jonah's observation, Elise, or the northbound train. |
| relationship_secret.location | the Ashcombe station hotel | F/F | F/F | F/F | The sampled station mention is an unrelated journey and does not identify the Ashcombe station hotel as the scene. |
| possession_scene_detail_goal.character_name | Mara Venn | F/F | F/F | F/F | No continuation names Mara Venn or maintains her as the active character. |
| possession_scene_detail_goal.object_possession | the lighthouse logbook wrapped in oilcloth | F/F | F/F | F/F | No output mentions the logbook, oilcloth, or Mara's possession of it. |
| possession_scene_detail_goal.location | Saint Orra lighthouse | F/F | F/F | F/F | No continuation identifies the setting as Saint Orra lighthouse. |
| possession_scene_detail_goal.scene_detail | the lower window is marked with a blue chalk circle | F/F | F/F | F/F | Blue color or entrance language is unrelated; no lower window, chalk circle, or safe-entry relation is preserved. |
| possession_scene_detail_goal.stated_goal | deliver the logbook to keeper Amos Bell | F/F | F/F | F/F | No output preserves delivery of the logbook to Amos Bell. |

## Final Questions

1. **Did contrastive training learn the pair objective?** Yes: held-out pairwise accuracy is 97.65% on test and 95.17% on generalization, versus 47.76% and 51.69% for Compute5.
2. **Did it transfer to narrative generation?** No: conservative manual scoring is 0/13 for greedy and sampled generation for Compute5, Narrative-v1, and Contrastive-v1; qualitative premise and entity-state continuity remain absent.
3. **Did the internal representation change broadly?** No broad change is demonstrated. Object-location decoding improved on both probe holdouts, but the other seven supported families are inconsistent or worse.

Context2k is consistent historical evidence: doubling context also retained 0/13 facts in both decoding modes. Context visibility alone did not solve this behavior.

Limitations: the synthetic pairs are templated and may admit lexical/template shortcuts; the frozen probe's lexical-rank labels are not invariant semantic roles.
