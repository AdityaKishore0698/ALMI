import os 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import json
import math
import clip
from torch.distributions import Categorical

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from dataset import dataset_CL_400sl
import models.almi_trans as trans
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)


##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Network ---- #####
clip_model, clip_preprocess = clip.load("pretrained/ViT-B-32.pt", device=torch.device(args.device), jit=False)
clip.model.convert_weights(clip_model) 
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

print(f'###########  train without tokenizer ##################')
trans_encoder = trans.ALMITransformer(num_obs=args.num_obs, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.seq_length+1, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate,
                                pred_action=args.pred_action)
if args.pred_action:
    print("pred only action")
else:
    print("pred obs-action pair")
##### ---- Optimization goals ---- #####
loss_mse = torch.nn.MSELoss()

train_loader = dataset_CL_400sl.DATALoader(args.dataname, args.batch_size, args.seq_length)
train_loader_iter = dataset_CL_400sl.cycle(train_loader)

# Gradient accumulation: ALMI_ACCUM_STEPS mini-batches of --batch-size make one optimizer step,
# so batch_size * accum reproduces the paper's batch when the GPU is too small for it.
# One "iteration" below is one optimizer step, as in the original script.
accum = int(os.environ.get("ALMI_ACCUM_STEPS", 1))
one_epoch_iters = len(train_loader) // accum
print(f'one epoch iters: {one_epoch_iters} (effective batch {args.batch_size * accum})')

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.to(args.device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

nb_iter, avg_loss, nb_epoch = 0, 0., 0
avg_loss_cls, avg_acc = 0., 0.
right_num = 0
nb_sample_train = 0

while nb_epoch <= args.total_epoch:
    if nb_epoch == 0:
        torch.save({'trans' : trans_encoder.state_dict()}, os.path.join(args.out_dir, f'almi_trans_cl_400sl_{nb_epoch}.pth'))
        logger.info(f'model saved in epoch {nb_epoch}')
    nb_epoch += 1
    for step in range(one_epoch_iters):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(accum):
            batch = next(train_loader_iter)
            clip_text, obs_actions, motion_len = batch
            bs = obs_actions.shape[0]

            history = obs_actions[:, :-1, :]
            if args.add_noise:
                pass
            target = obs_actions
            target, history = target.to(args.device), history.to(args.device)

            text = clip.tokenize(clip_text, truncate=True).to(args.device)
            feat_clip_text = clip_model.encode_text(text).float()

            pred_obs_action = trans_encoder(history, feat_clip_text)

            loss = 0.0
            for i in range(bs):
                if args.pred_action:
                    loss += loss_mse(pred_obs_action[i][:motion_len[i] + 1], target[i][:motion_len[i] + 1, -23:-2]) / bs
                else:
                    loss += loss_mse(pred_obs_action[i][:motion_len[i] + 1], target[i][:motion_len[i] + 1]) / bs

            ## global loss, averaged over the accumulated mini-batches
            (loss / accum).backward()
            step_loss += loss.item() / accum
        optimizer.step()
        scheduler.step()

        avg_loss = avg_loss + step_loss

        nb_iter += 1
        if nb_iter % args.print_iter ==  0 :
            avg_loss = avg_loss / args.print_iter
            writer.add_scalar('./Loss/train', avg_loss, nb_iter)
            msg = f"Train. Iter {nb_iter} : Loss. {avg_loss:.5f}"
            logger.info(msg)
            avg_loss = 0.

    # save model
    if nb_epoch % int(os.environ.get("ALMI_SAVE_EVERY_EPOCHS", 10)) == 0 :
        torch.save({'trans' : trans_encoder.state_dict()}, os.path.join(args.out_dir, 'almi_trans_cl_400sl_last.pth'))
        logger.info(f'model last saved {nb_epoch}')
