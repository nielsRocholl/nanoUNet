"""In-process golden stages C (dataloaders), D (model/loss/opt/LR/EMA/MAE), E (state keys),
H (old-ckpt load). `python equiv/_stage.py <C|D|E|H> <out.json> [fixtures_dir]`; env from capture.py.

Every stage re-seeds random/np/torch to 0 before each sub-step (K12: aug workers and the MAE mask
draw from the global RNGs), so sub-steps are independent of each other's draw counts."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _boot  # noqa: E402
from _digest import digest  # noqa: E402

DS = "Dataset903_Merged"
PLANS = "nnUNetResEncUNetLPlans"


def _paths():
    pp, raw = os.environ["NANOUNET_PREPROCESSED"], os.environ["NANOUNET_RAW"]
    extra = os.path.join(os.path.dirname(pp), "extra")
    return (os.path.join(pp, DS, PLANS + ".json"), os.path.join(raw, DS, "dataset.json"),
            os.path.join(extra, "configs"), os.path.join(pp, DS, "valset_small.json"), os.path.dirname(pp))


def _dm(cfg, nw, **kw):
    from nanounet.dataloader_prefs import DataloaderBucket
    from nanounet.train.data_module import NanoDataModule

    dm = NanoDataModule(DS, 0, PLANS, cfg, DataloaderBucket(nw, nw, 2, 2), None, kw.pop("iters", 6), 2, **kw)
    dm.setup("fit")
    return dm


def _batches(dl, n=None):
    out = []
    for i, b in enumerate(dl):
        if n is not None and i >= n:
            break
        out.append(b)
    return out


def stage_c(res):
    from nanounet.dataloader_prefs import DataloaderBucket
    from nanounet.pretrain.dataset import build_pretrain_dataloaders

    _, _, cfgs, vman, _ = _paths()
    for cname in ("default", "instance_conditional"):
        for nw in (0, 2):
            _boot.seed_all(0)
            dm = _dm(os.path.join(cfgs, cname + ".json"), nw)
            _boot.seed_all(0)
            for i, b in enumerate(_batches(dm.train_dataloader())):
                res[f"C/{cname}/nw{nw}/train/{i}"] = digest(b)
            _boot.seed_all(0)
            for i, b in enumerate(_batches(dm.val_dataloader())):
                res[f"C/{cname}/nw{nw}/val/{i}"] = digest(b)
    for nw in (0, 2):
        _boot.seed_all(0)
        dm = _dm(os.path.join(cfgs, "default.json"), nw, val_manifest=vman)
        _boot.seed_all(0)
        for i, b in enumerate(_batches(dm.val_dataloader())):
            res[f"C/manifest/nw{nw}/val/{i}"] = digest(b)
        _boot.seed_all(0)
        dm2 = _dm(os.path.join(cfgs, "default.json"), nw, prompts_per_patch=2, iters=2)
        _boot.seed_all(0)
        for i, b in enumerate(_batches(dm2.train_dataloader())):
            res[f"C/ppp2/nw{nw}/train/{i}"] = digest(b)
        _boot.seed_all(0)
        tr, va = build_pretrain_dataloaders(DS, 0, PLANS, 2, 4, 2, 3000, 4000, DataloaderBucket(nw, nw, 2, 2),
                                            persistent_workers=False)
        _boot.seed_all(0)
        for i, b in enumerate(_batches(tr)):
            res[f"C/mae/nw{nw}/train/{i}"] = digest(b)
        _boot.seed_all(0)
        for i, b in enumerate(_batches(va)):
            res[f"C/mae/nw{nw}/val/{i}"] = digest(b)


def _grads(net):
    return {n: p.grad for n, p in sorted(net.named_parameters()) if p.grad is not None}


def stage_d(res):
    import torch

    from nanounet.model.losses import consistency_dice_term
    from nanounet.model.lr_schedule import PolyLRScheduler, StretchedTailPolyLRScheduler
    from nanounet.pretrain.dataset import build_pretrain_dataloaders
    from nanounet.dataloader_prefs import DataloaderBucket
    from nanounet.pretrain.module import NanoMAELM
    from nanounet.train.ema import EMACallback
    from nanounet.train.lightning_module import NanoUNetLM

    dh = _boot.mod("nanounet.model.dice_helpers")
    plans, dj, cfgs, _, root = _paths()
    cfg = os.path.join(cfgs, "default.json")
    out_dir = os.path.join(root, "results", "D")
    _boot.seed_all(0)
    batch = _batches(_dm(cfg, 0, iters=1).train_dataloader())[0]
    _boot.seed_all(0)
    pbatch = _batches(_dm(cfg, 0, prompts_per_patch=2, iters=1).train_dataloader())[0]
    _boot.seed_all(0)
    vbatch = _batches(_dm(cfg, 0).val_dataloader())[0]
    for loss_type in ("dc_ce", "cc_dc_ce"):
        _boot.seed_all(0)
        lm = NanoUNetLM(plans, dj, cfg, out_dir, loss_type=loss_type)
        res[f"D/{loss_type}/init"] = digest(lm.net.state_dict())
        out = lm.net(batch["data"])
        loss = lm.loss(out, batch["target"])
        res[f"D/{loss_type}/out"] = digest(out)
        res[f"D/{loss_type}/loss"] = digest(loss.detach())
        loss.backward()
        res[f"D/{loss_type}/grads"] = digest(_grads(lm.net))
        opt = lm.configure_optimizers()["optimizer"]
        opt.step()
        res[f"D/{loss_type}/step"] = digest(lm.net.state_dict())
    _boot.seed_all(0)
    lm = NanoUNetLM(plans, dj, cfg, out_dir, consistency_weight=0.5)
    out = lm.net(pbatch["data"])
    lc = consistency_dice_term(out, pbatch["pair_id"])
    tot = lm.loss(out, pbatch["target"]) + 0.5 * lc
    tot.backward()
    res["D/consistency/loss"] = digest([lc.detach(), tot.detach()])
    res["D/consistency/grads"] = digest(_grads(lm.net))
    with torch.no_grad():
        o1 = lm.net(vbatch["data"])
        o2 = lm.net(vbatch["data_prompt2"])
        lv = lm.loss(o1, vbatch["target"])
        row = dh.val_step_row(o1, vbatch["target"], lm.label_manager, True, float(lv), vbatch["click_inside"])
        res["D/val/row"] = digest(row)
        res["D/val/pooled"] = digest(dh.pooled_fg_dice([row]))
        res["D/val/pair"] = digest(dh.prompt_pair_dice(o1, o2, True))
        res["D/val/click_split"] = digest(dh.click_split_means([row]))
    for name, mk in (
        ("poly", lambda o: PolyLRScheduler(o, 0.01, 20)),
        ("poly_warm", lambda o: PolyLRScheduler(o, 0.01, 20, warmup_epochs=3)),
        ("stretched", lambda o: StretchedTailPolyLRScheduler(o, 0.01, 20, k_transition=15, ref_poly_steps=20,
                                                             exponent=0.9, warmup_epochs=2)),
    ):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        s = mk(opt)
        lrs = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"])
            s.step()
        res[f"D/lr/{name}"] = digest(lrs)
    _boot.seed_all(0)
    lm = NanoUNetLM(plans, dj, cfg, out_dir)
    opt = lm.configure_optimizers()["optimizer"]
    ema = EMACallback(decay=0.999)
    for _ in range(3):
        opt.zero_grad()
        lm.loss(lm.net(batch["data"]), batch["target"]).backward()
        opt.step()
        ema.on_train_batch_end(None, lm, None, batch, 0)
    res["D/ema/shadow"] = digest(ema.state_dict())
    _boot.seed_all(0)
    tr, _ = build_pretrain_dataloaders(DS, 0, PLANS, 2, 1, 1, 3000, 4000, DataloaderBucket(0, 0, 2, 2))
    mb = _batches(tr)[0]
    _boot.seed_all(0)
    mae = NanoMAELM(plans, dj, os.path.join(root, "results", "Dmae"))
    res["D/mae/init"] = digest(mae.net.state_dict())
    ml = mae._loss(mb)
    ml.backward()
    res["D/mae/loss"] = digest(ml.detach())
    res["D/mae/grads"] = digest(_grads(mae.net))


def _keys(sd):
    return [(k, list(v.shape), str(v.dtype)) for k, v in sd.items()]


def stage_e(res):
    from nanounet.pretrain.module import NanoMAELM
    from nanounet.train.lightning_module import NanoUNetLM

    plans, dj, cfgs, _, root = _paths()
    _boot.seed_all(0)
    lm = NanoUNetLM(plans, dj, os.path.join(cfgs, "default.json"), os.path.join(root, "results", "E"))
    mae = NanoMAELM(plans, dj, os.path.join(root, "results", "Emae"))
    res["E/sup/keys"] = digest(_keys(lm.state_dict()))
    res["E/mae/keys"] = digest(_keys(mae.state_dict()))
    res["E/sup/hparams"] = digest(dict(lm.hparams))
    res["E/mae/hparams"] = digest(dict(mae.hparams))
    res["E/sup/keylist"] = sorted(lm.state_dict())[:3] + [len(lm.state_dict())]


def stage_h(res, fx):
    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json

    from nanounet.infer.predictor import load_net_from_ckpt
    from nanounet.lightning_ckpt import pl_ckpt_epoch_and_target
    from nanounet.model.mae_transfer import load_full_net, load_mae_encoder
    from nanounet.model.network import build_net
    from nanounet.plan.plans import Plans

    plans, dj_path, _, _, _ = _paths()
    sup, mae = os.path.join(fx, "old_sup.ckpt"), os.path.join(fx, "old_mae.ckpt")
    pm = Plans(plans)
    cm, dj = pm.get_configuration("3d_fullres"), load_json(dj_path)
    for ema in (False, True):
        net, lm = load_net_from_ckpt(sup, cm, dj, torch.device("cpu"), ema=ema)
        res[f"H/load_net/ema{int(ema)}"] = digest(net.state_dict())
    for name, fn, ck in (("mae_encoder", load_mae_encoder, mae), ("full_net", load_full_net, sup)):
        _boot.seed_all(0)
        net = build_net(cm, pm.get_label_manager(dj), dj, True)
        info = fn(net, ck)
        res[f"H/{name}/info"] = digest([sorted(info["loaded"]), list(info["missing"]), list(info["unexpected"])])
        res[f"H/{name}/state"] = digest(net.state_dict())
    res["H/epoch_target"] = digest([pl_ckpt_epoch_and_target(sup), pl_ckpt_epoch_and_target(mae)])


if __name__ == "__main__":
    _boot.pin()
    stage, out = sys.argv[1], sys.argv[2]
    _boot.check_src()
    res: dict = {}
    if stage == "C":
        stage_c(res)
    elif stage == "D":
        stage_d(res)
    elif stage == "E":
        stage_e(res)
    elif stage == "H":
        stage_h(res, sys.argv[3])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, sort_keys=True)
