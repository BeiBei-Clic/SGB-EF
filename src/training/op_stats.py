def count_ops_from_mask(u_mask_x, vocab_size):
    """统计操作分布，返回(ins, del, sub, keep, valid)。"""
    valid = (u_mask_x.sum(dim=-1) > 0)
    if valid.sum() == 0:
        return 0, 0, 0, 0, 0

    target_ids = u_mask_x.argmax(dim=-1)
    target_ids = target_ids[valid]

    ins = (target_ids < vocab_size).sum().item()
    delete = (target_ids == vocab_size).sum().item()
    sub = ((target_ids > vocab_size) & (target_ids < 2 * vocab_size + 1)).sum().item()
    keep = (target_ids == 2 * vocab_size + 1).sum().item()

    return ins, delete, sub, keep, target_ids.numel()


def compute_inverse_weights(ins, delete, sub, keep, eps=1e-6):
    total = ins + delete + sub + keep
    if total <= 0:
        return {"ins": 1.0, "del": 1.0, "sub": 1.0, "keep": 1.0}

    w_ins = total / (ins + eps)
    w_del = total / (delete + eps)
    w_sub = total / (sub + eps)
    w_keep = total / (keep + eps)

    avg = (w_ins + w_del + w_sub + w_keep) / 4.0
    return {
        "ins": w_ins / avg,
        "del": w_del / avg,
        "sub": w_sub / avg,
        "keep": w_keep / avg,
    }
