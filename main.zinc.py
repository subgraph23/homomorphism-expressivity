from src.utils import *
from src.model import *
from functools import partial

# --------------------------------- ARGPARSE --------------------------------- #

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="type of GNN layer")
parser.add_argument('--full', dest="subset", action="store_false", help="run full ZINC")
parser.add_argument('--subset', dest="subset", action="store_true", help="only subset")
parser.add_argument("--test", action="store_true", dest="test")

parser.add_argument("--seed", type=int, default=19260817, help="random seed")
parser.add_argument("--indir", type=str, default="data/zinc", help="dataset")
parser.add_argument("--outdir", type=str, default="result.zinc", help="output")
parser.add_argument("--device", type=int, default=None, help="CUDA device")

parser.add_argument("--max_dis", type=int, default=5, help="distance encoding")
parser.add_argument("--num_layer", type=int, default=6, help="number of layers")
parser.add_argument("--dim_embed", type=int, default=96, help="embedding dimension")

parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=400, help="training epochs")

args = parser.parse_args()
print(f"""Run:
    model: {args.model}
    subset: {args.subset}
    seed: {args.seed}
""")

if args.model == "MP":
    args.dim_embed *= 11
    args.dim_embed //= 7
    args.max_dis = 0
elif args.model == "Sub-G":
    args.dim_embed *= 5
    args.dim_embed //= 4
        

id = f"{args.model}-{args.subset}-{args.max_dis}-{args.num_layer}x{args.dim_embed}-{args.bs}-{args.lr}-{args.seed}"

torch.manual_seed(args.seed)
if args.device is None: device = torch.device("cpu") 
else: device = torch.device(f"cuda:{args.device}") 

# ---------------------------------- DATASET --------------------------------- #

from src import dataset

dataloader = {
    name: data.dataloader.DataLoader(
        dataset.ZINC(args.indir,
                     args.subset, name,
                     transform=subgraph),
        batch_size=args.bs,
        num_workers=4,
        shuffle=True
    )
    for name in ["train", "val", "test"]
}

# ----------------------------------- MODEL ---------------------------------- #
# copied from OGB mol_encoder.py

class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_atom_feature_dims):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])
        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim, full_bond_feature_dims=None):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])
        return bond_embedding   
    
    
model = GNN(args, "g",
            enc_a=partial(AtomEncoder, full_atom_feature_dims=[32]),
            enc_e=partial(BondEncoder, full_bond_feature_dims=[5]),
            odim=1)

# ----------------------------------- ITER ----------------------------------- #

def critn(pred, batch):
    pred = pred.squeeze(-1)
    assert pred.shape == (y:=batch.y).shape
    return torch.nn.L1Loss()(pred, y)

def train(model, loader, critn, optim):
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        optim.zero_grad()
        loss = critn(pred, batch)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return np.array(losses).mean()

def eval(model, loader, critn):
    model.eval()
    errors = []
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
          pred = model(batch)
          err = critn(pred, batch)
          errors.append(err.item())
    return np.array(errors).mean()

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                   mode='min',
                                                   factor=0.5,
                                                   patience=20,
                                                #    min_lr=1e-5,
                                                   verbose=True
                                                   )

if args.test:
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.numel())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    import code;
    exit(code.interact(local=dict(globals(), **dict(locals()))))

# ------------------------------------ RUN ----------------------------------- #

record = []

os.makedirs(output_dir:=args.outdir, exist_ok=True)
assert not os.path.exists(log:=f"{output_dir}/{id}.txt")

from tqdm import trange
for epoch in (pbar:=trange(args.epochs)):
    lr = optim.param_groups[0]['lr']
    loss = train(model, dataloader["train"], critn, optim)
    val = eval(model, dataloader["val"], critn)
    test = eval(model, dataloader["test"], critn)

    sched.step(val)

    record.append([lr, loss, val, test])
    pbar.set_postfix(lr=lr, loss=loss, val=val, test=test)

    np.savetxt(log, record, delimiter='\t')
