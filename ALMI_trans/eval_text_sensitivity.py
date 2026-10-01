"""Does a CL-20sl-type model use the hand part of its text command?

On random 21-step windows of the waving training data, the model's next-action error is compared
for the true caption, the caption with the hand word swapped (left<->right, both->left) and the
caption without the hand phrase. If the model follows the hand instruction, swapping the hand word
must raise the arm error. Also prints how far the predicted arm actions move when only the hand
word changes. For chunked models the first action of each chunk is used.

Usage (from ALMI_trans/, needs the GPU): python eval_text_sensitivity.py <checkpoint.pth>
"""
import random, re, sys
import numpy as np, torch, clip
sys.path.insert(0, ".")
import models.almi_trans as trans
from dataset import dataset_CL_20sl

torch.manual_seed(0); random.seed(0)
dev = "cuda:0"
ckpt = sys.argv[1]
state = torch.load(ckpt, map_location="cpu")["trans"]
model = trans.ALMITransformer(num_obs=71, embed_dim=256, clip_dim=512, block_size=21, num_layers=9, n_head=16,
                              drop_out_rate=0.1, fc_rate=4, pred_action=True, chunk=trans.chunk_from_state(state))
trans.load_trans_state(model, state); model.eval().to(dev)
clip_model, _ = clip.load("pretrained/ViT-B-32.pt", device=dev, jit=False); clip_model.eval()
ds = dataset_CL_20sl.ALMI_CL_20slDataset("almi")
SWAP = {"left": "right", "right": "left", "both": "left"}
ARM = slice(13, 21)   # action dims: 12 legs, waist, 4 left arm, 4 right arm
LARM, RARM = slice(13, 17), slice(17, 21)

def enc(texts):
    with torch.no_grad():
        return clip_model.encode_text(clip.tokenize(texts, truncate=True).to(dev)).float()

N = 2000
names = random.sample(ds.name_list, min(N, len(ds.name_list)))
hist, tgt, true_t, swap_t, none_t, hands = [], [], [], [], [], []
for n in names:
    d = ds.data_dict[n]; oa = d["obs_actions"]; cap = d["text"][0]["caption"]
    if len(oa) < 22: continue
    s = random.randint(0, len(oa) - 21)
    seg = oa[s:s + 21]
    h = re.search(r"wave (\w+)", cap).group(1)
    hist.append(seg[:-1]); tgt.append(seg[1:, -23:-2]); hands.append(h)
    true_t.append(cap); swap_t.append(cap.replace(f"wave {h}", f"wave {SWAP[h]}"))
    none_t.append(re.sub(r" and wave \w+( hands?)?", "", cap))
hist = torch.tensor(np.stack(hist)).to(dev); tgt = torch.tensor(np.stack(tgt)).to(dev)
print("examples:", true_t[0], "|", swap_t[0], "|", none_t[0])

def err(texts):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            p = model(hist[i:i+256], enc(texts[i:i+256]))[:, 1:, :21]
            out.append(((p - tgt[i:i+256]) ** 2))
    return torch.cat(out)  # (N, 20, 21)

e_true, e_swap, e_none = err(true_t), err(swap_t), err(none_t)
for name, sl in (("arms", ARM), ("left arm", LARM), ("right arm", RARM), ("legs", slice(0, 12))):
    print(f"{name:9s} MSE by history length  true / swapped hand / no hand phrase")
    for pos in (0, 1, 4, 9, 19):
        print(f"   history {pos+1:2d}: {e_true[:, pos, sl].mean():.5f}  {e_swap[:, pos, sl].mean():.5f}  {e_none[:, pos, sl].mean():.5f}"
              f"   ratio swap/true {e_swap[:, pos, sl].mean() / e_true[:, pos, sl].mean():.2f}")
# how much the predicted arm action moves when only the hand word changes, vs how much it moves per step
with torch.no_grad():
    pa = torch.cat([model(hist[i:i+256], enc(true_t[i:i+256]))[:, 1:, :21] for i in range(0, len(true_t), 256)])
    pb = torch.cat([model(hist[i:i+256], enc(swap_t[i:i+256]))[:, 1:, :21] for i in range(0, len(swap_t), 256)])
step = (tgt[:, 1:, ARM] - tgt[:, :-1, ARM]).abs().mean()
print(f"mean |arm action change| from swapping the hand word: {(pa - pb)[:, :, ARM].abs().mean():.4f} "
      f"(position 1: {(pa - pb)[:, 0, ARM].abs().mean():.4f}, position 20: {(pa - pb)[:, -1, ARM].abs().mean():.4f}); "
      f"typical per-step change in the data: {step:.4f}; arm action std in data: {tgt[:, :, ARM].std():.4f}")
