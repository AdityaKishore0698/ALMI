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
from dataset import dataset_pre_OL_VQ
from dataset import dataset_OL

import models.t2m_trans as trans
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir = os.path.join("./dataset/ALMI", f"{args.vq_name}")  # where dataset_OL.py reads the tokens
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- VQ dataloader ---- #####
train_loader_token = dataset_pre_OL_VQ.DATALoader(args.dataname, 1, unit_length=2**args.down_t, max_motion_len=args.max_motion_len)

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

net = vqvae.VQVAE(args,
                args.nb_code,
                args.code_dim,
                args.output_emb_width,
                args.down_t,
                args.stride_t,
                args.width,
                args.depth,
                args.dilation_growth_rate,
                args.vq_act,)

trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net_vq'], strict=True)
net.eval()
net.to(args.device)

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.to(args.device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss, nb_epoch = 0, 0., 0
avg_loss_cls, avg_acc = 0., 0.
right_num = 0
nb_sample_train = 0

##### ---- get code ---- #####
for batch in train_loader_token:
    pose, name = batch

    pose = pose.to(args.device).float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose).squeeze()
    target = target.cpu().numpy()
    np.save(pjoin(args.vq_dir, name[0] +'.npy'), target)

# if args.pred_all:
train_loader = dataset_OL.DATALoader(args.dataname, args.batch_size, args.nb_code, 
                                                args.vq_name, unit_length=2**args.down_t, 
                                                use_tcn=args.use_tcn, max_motion_len=args.max_motion_len)
# Gradient accumulation: ALMI_ACCUM_STEPS mini-batches of --batch-size make one optimizer step,
# so batch_size * accum reproduces the paper's batch when the GPU is too small for it.
accum = int(os.environ.get("ALMI_ACCUM_STEPS", 1))
one_epoch_iters = len(train_loader) // accum
train_loader_iter = dataset_OL.cycle(train_loader)
print(f'one epoch iters: {one_epoch_iters} (effective batch {args.batch_size * accum})')

##### ---- Training ---- #####
while nb_epoch <= args.total_epoch:
    if nb_epoch == 0:
        torch.save({'trans' : trans_encoder.state_dict()}, os.path.join(args.out_dir, f'almi_trans_ol_{nb_epoch}.pth'))
        logger.info(f'model saved in epoch {nb_epoch}')
    train_loader_iter = iter(train_loader)
    nb_epoch += 1
    for step in range(one_epoch_iters):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(accum):
            batch = next(train_loader_iter)
            clip_text, m_tokens, m_tokens_len = batch
            m_tokens, m_tokens_len = m_tokens.to(args.device), m_tokens_len.to(args.device)
            bs = m_tokens.shape[0]
            target = m_tokens    # (bs, 26)
            target = target.to(args.device)
        
            text = clip.tokenize(clip_text, truncate=True).to(args.device)
            feat_clip_text = clip_model.encode_text(text).float()

            input_index = target[:,:-1]

            if args.pkeep == -1:  # pkeep 1.0
                proba = np.random.rand(1)[0]
                mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                                device=input_index.device))
            else:
                mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                                device=input_index.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(input_index, args.nb_code)
            a_indices = mask*input_index+(1-mask)*r_indices

            cls_pred = trans_encoder(a_indices, feat_clip_text)
            cls_pred = cls_pred.contiguous()

            loss = 0.0
            for i in range(bs):
                # loss function     (26), (26, 513)
                loss += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs

                # Accuracy
                probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)

                if args.if_maxtest:
                    _, cls_pred_index = torch.max(probs, dim=-1)

                else:
                    dist = Categorical(probs)
                    cls_pred_index = dist.sample()
                right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()

            ## global loss, averaged over the accumulated mini-batches
            (loss / accum).backward()
            step_loss += loss.item() / accum
            nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()
        optimizer.step()
        scheduler.step()

        avg_loss_cls = avg_loss_cls + step_loss

        nb_iter += 1
        if nb_iter % args.print_iter ==  0 :
            avg_loss_cls = avg_loss_cls / args.print_iter
            avg_acc = right_num * 100 / nb_sample_train
            writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
            writer.add_scalar('./ACC/train', avg_acc, nb_iter)
            msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
            logger.info(msg)
            avg_loss_cls = 0.
            right_num = 0
            nb_sample_train = 0

    # save model
    if nb_epoch % int(os.environ.get("ALMI_SAVE_EVERY_EPOCHS", 10)) == 0 :
        torch.save({'trans' : trans_encoder.state_dict()}, os.path.join(args.out_dir, 'almi_trans_ol_last.pth'))
        logger.info(f'model last saved {nb_epoch}')
