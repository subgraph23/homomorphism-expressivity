from src.utils import *
from src.model import *

# --------------------------------- ARGPARSE --------------------------------- #

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True, help="type of GNN layer")
parser.add_argument("--task", type=str, nargs=2, required=True, help="name&task")
parser.add_argument("--test", action="store_true", dest="test")

parser.add_argument("--seed", type=int, default=19260817, help="random seed")
parser.add_argument("--indir", type=str, default="data/count", help="dataset")
parser.add_argument("--outdir", type=str, default="result.count", help="output")
parser.add_argument("--device", type=int, default=None, help="CUDA device")

parser.add_argument("--max_dis", type=int, default=5, help="distance encoding")
parser.add_argument("--num_layer", type=int, default=5, help="number of layers")
parser.add_argument("--dim_embed", type=int, default=96, help="embedding dimension")

parser.add_argument("--bs", type=int, default=512, help="batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=1200, help="training epochs")

args = parser.parse_args()
print(f"""Run:
    model: {args.model}
    task: {args.task}
    seed: {args.seed}
""")

if args.model in ["MP", "Sub"]:
    args.dim_embed  *= 4
    args.dim_embed //= 3
    if args.model == "MP":
        args.max_dis = 0

id = f"{args.model}-{args.task}-{args.max_dis}-{args.num_layer}x{args.dim_embed}-{args.bs}-{args.lr}-{args.seed}"

torch.manual_seed(args.seed)
if args.device is None: device = torch.device("cpu") 
else: device = torch.device(f"cuda:{args.device}") 

# ---------------------------------- DATASET --------------------------------- #

from src import dataset

dataloader = {
    name: data.dataloader.DataLoader(
        dataset.GraphCount(args.indir,
                           name, *args.task,
                           transform=subgraph),
        follow_batch=["y"],
        batch_size=args.bs,
        num_workers=4,
        shuffle=True
    )
    for name in ["train", "val", "test"]
}

# ----------------------------------- MODEL ---------------------------------- #

model = GNN(args, args.task[1], enc_a=None, enc_e=None, odim=1, bn=False)

# ----------------------------------- ITER ----------------------------------- #

def critn(pred, batch):

    pred = pred.squeeze(-1)
    assert pred.shape == (y:=batch.y).shape

    L1 = torch.abs(pred - y)
    
    normalize = {
        "cycle3,v": 3, "cycle3,e": 6,
        "cycle4,v": 4, "cycle4,e": 8,
        "cycle5,v": 5, "cycle5,e": 10,
        "cycle6,v": 6, "cycle6,e": 12,
        "chordal4,v": 4, "chordal4,e": 10,
        "chordal5,v": 5, "chordal5,e": 14,
    }
    task = f"{args.task[0]},{args.task[1]}"
    if task in normalize:
        L1 = L1 / normalize[task]

    if args.task[1] == "e":
       L1 = L1 * batch["e"]

    return pys.scatter(L1, batch["y_batch"]).mean()

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
                                                   factor=0.9,
                                                   patience=10,
                                                #    min_lr=1e-5,
                                                   verbose=True
                                                   )

if args.test:
    # print(gnn.summary(model, next(iter(dataloader["train"])), max_depth=5))
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name, p.numel(), p.std().item())
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    import code;
    exit(code.interact(local=dict(globals(), **dict(locals()))))

# ------------------------------------ RUN ----------------------------------- #

record = []

os.makedirs(output_dir:=args.outdir, exist_ok=True)
assert not os.path.exists(log:=f"{output_dir}/{id}.txt")

from tqdm import trange
for epoch in (pbar:=trange(args.epochs)):

    # for group in optim.param_groups:
    #     group['lr'] = args.lr * (1 + np.cos(np.pi * epoch / args.epochs))/2
    lr = optim.param_groups[0]['lr']
    loss = train(model, dataloader["train"], critn, optim)
    test = eval(model, dataloader["test"], critn)
    val = eval(model, dataloader["val"], critn)
    
    sched.step(val)

    record.append([lr, loss, val, test])
    pbar.set_postfix(lr=lr, loss=loss, val=val, test=test)

    np.savetxt(log, record, delimiter='\t')

