import torch
import argparse
import models.almi_trans as trans
import copy
import os

class PolicyExporterTrans(torch.nn.Module):
    def __init__(self, trans_model):
        super().__init__()
        self.trans_base = copy.deepcopy(trans_model.trans_base)
        self.trans_head = copy.deepcopy(trans_model.trans_head)

    def forward(self, idxs, clip_feature):
        feat = self.trans_base(idxs, clip_feature)
        out = self.trans_head(feat)
        return out

    def export(self, path, ckpt):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f'policy_trans_{ckpt}.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def main(args):

    # load trans model
    his_max_len = args.seq_len
    ckpt = torch.load(args.model_path, map_location='cpu')
    chunk = trans.chunk_from_state(ckpt['trans'])
    trans_model = trans.ALMITransformer(num_obs=71, 
                            embed_dim=256, 
                            clip_dim=512, 
                            block_size=his_max_len+1, 
                            num_layers=9,
                            n_head=16,
                            drop_out_rate=0.1, 
                            fc_rate=4,
                            pred_action=True,
                            chunk=chunk)
    trans.load_trans_state(trans_model, ckpt['trans'])
    print(f"action chunk {chunk}, text normalisation {'on' if 'trans_base.text_mean' in ckpt['trans'] else 'off'}")
    # trans_model.to("cuda:0")
    total_params = sum(p.numel() for p in trans_model.parameters())
    print(f"successfully load trans model, total params {total_params}")

    exporter = PolicyExporterTrans(trans_model)
    exporter.export(args.export_path, args.export_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--export_path', type=str)
    parser.add_argument('--export_name', type=str)
    args = parser.parse_args()
    main(args)