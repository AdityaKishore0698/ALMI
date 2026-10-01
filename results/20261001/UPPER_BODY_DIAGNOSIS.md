# Why CL-20sl waves the wrong hand

CL-20sl follows the locomotion part of a command but waves the instructed hand in only one of three
Table 13 commands (paper: 20–40 %). Before changing the model, we checked whether it uses the hand part
of the text at all.

## Test

`ALMI_trans/eval_text_sensitivity.py` takes 2,000 random 21-step windows of the waving training data and
compares the model's next-action error for three versions of each caption:

* the true caption ("Robot go left and wave left."),
* the hand word swapped: left ↔ right, both → left ("Robot go left and wave right."),
* the hand phrase removed ("Robot go left.").

A model that follows the hand instruction must predict the arms worse when the hand word is wrong.

## Result (baseline CL-20sl, `cl20sl_20260929_ae45cca_r1`)

Arm (waist excluded) mean squared error of the next action:

| History length | True caption | Hand swapped | Hand phrase removed | Swapped / true |
|---|---|---|---|---|
| 1 | 0.00055 | 0.00056 | 0.00071 | 1.02 |
| 5 | 0.00047 | 0.00048 | 0.00060 | 1.03 |
| 20 | 0.00038 | 0.00038 | 0.00050 | 1.00 |

By arm, with 20 steps of history: left 1.07, right 0.93 (the right arm is predicted slightly *better*
with the wrong hand word).

Swapping the hand word moves the predicted arm actions by 0.0038 on average. One step of motion in the
data moves them by 0.051, 13 times more. The model reads that it should wave (removing the phrase costs
about 30 %), but not which hand.

## Cause

CLIP (ViT-B/32) text features, cosine similarity:

| | wave left | wave right | wave both | left fast, wave left | backward slowly, wave left | forward fast (no wave) |
|---|---|---|---|---|---|---|
| forward fast, wave left | 1 | 0.996 | 0.996 | 0.979 | 0.974 | 0.973 |

The three hand variants of a command are closer to each other (0.995–0.996) than any two locomotion
commands (0.93–0.98). The single text token the Transformer receives therefore carries almost no
hand information, and the 20-step history already predicts the arms well, so training never has to
use it.

## Experiments started (2026-10-01 21:53)

Both runs use the paper's CL-20sl settings and the baseline's 126k optimizer steps, so differences
come from the change alone.

* **A, text-feature standardisation** (`--text-norm`): each CLIP dimension is standardised with the mean
  and standard deviation over the 123 distinct training captions, so that the hand variants differ as
  much as the locomotion commands do. Stored in the checkpoint and applied in the exported policy.
* **B, A + action chunks of 10** (`--chunk 10`): each step predicts the next 10 actions, executed with
  temporal ensembling (`play_trans.py --ensemble_m`). A future arm motion cannot be read from the history,
  so the model has to take it from the text.

Measures: the sensitivity test above, then MuJoCo survival, velocity and hand success on the Table 13
commands.
