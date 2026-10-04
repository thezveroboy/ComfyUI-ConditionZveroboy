# nodes.py — ComfyUI-ConditionZveroboy
# Универсальные ноды для CONDITIONING (любой DiT: Krea2, Qwen-Image, Minimax и т.д.)
# Работают на уровне torch-тензоров (B, seq, dim), без знания конкретной модели.
# Проверка совместимости: batch и dim должны совпадать, иначе ошибка
# (так ловится попытка смешать Krea2 dim=30720 с Qwen другой dim).

import os
import torch
import torch.nn.functional as F

try:
    import folder_paths
    _HAS_FOLDER_PATHS = True
except Exception:
    folder_paths = None
    _HAS_FOLDER_PATHS = False


# ---------- dynamic inputs helper (как в TextJoinZveroboy) ----------

class ContainsAnyDict(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return dict.get(self, key)


COND_PREFIX = "cond_"
BASE_INPUTS = 2
EPS = 1e-8


def _collect_conds(prefix, fixed_first, fixed_second, kwargs):
    """Собрать cond_1, cond_2, ... cond_N из fixed + dynamic kwargs по номеру."""
    vals = [fixed_first, fixed_second]
    dyn = {k: v for k, v in kwargs.items()
           if isinstance(k, str) and k.startswith(prefix)}
    # cond_1/cond_2 уже есть в fixed, из kwargs их не дублируем
    def _num(k):
        try:
            return int(k.split("_", 1)[1])
        except Exception:
            return 10 ** 9
    for k in sorted(dyn.keys(), key=_num):
        n = _num(k)
        if n in (1, 2):
            continue
        vals.append(dyn[k])
    # выкинуть неподключенные
    return [v for v in vals if v is not None]


def _single_tensor(cond):
    """Из CONDITIONING достать единый тензор (B,L,D) + базовый dict.
    Если записей несколько (после Combine) — склеиваем по seq.
    """
    if cond is None or len(cond) == 0:
        raise ValueError("пустой CONDITIONING: вход не подключен")
    tensors = []
    base_dict = {}
    for i, entry in enumerate(cond):
        t, d = entry[0], entry[1] if len(entry) > 1 else {}
        if not torch.is_tensor(t):
            raise ValueError(f"запись {i}: не тензор")
        if t.dim() != 3:
            raise ValueError(f"запись {i}: ожидался тензор (B,seq,dim), получен {tuple(t.shape)}")
        tensors.append(t)
        if i == 0 and isinstance(d, dict):
            base_dict = dict(d)
    if len(tensors) == 1:
        return tensors[0], base_dict
    # несколько записей: требуем одинаковые B и dim, клеим по seq
    b0, d0 = tensors[0].shape[0], tensors[0].shape[2]
    for t in tensors[1:]:
        if t.shape[0] != b0 or t.shape[2] != d0:
            raise ValueError(
                f"внутри одного CONDITIONING разные B/dim: {tuple(tensors[0].shape)} vs {tuple(t.shape)}"
            )
    return torch.cat(tensors, dim=1), base_dict


def _align(tensors):
    """Проверить B/dim, выровнять seq нулевым паддингом вправо. Возврат (B, Lmax, D)."""
    if len(tensors) < 1:
        raise ValueError("нужен хотя бы один кондишен")
    b0 = tensors[0].shape[0]
    d0 = tensors[0].shape[2]
    for i, t in enumerate(tensors):
        if t.shape[0] != b0 or t.shape[2] != d0:
            raise ValueError(
                f"несовместимые кондишены (разные модели?): вход {i+1} имеет {tuple(t.shape)}, "
                f"а вход 1 — {tuple(tensors[0].shape)}. "
                f"Смешивать можно только кондишены одной модели/энкодера."
            )
    lmax = max(t.shape[1] for t in tensors)
    out = []
    for t in tensors:
        if t.shape[1] < lmax:
            pad = lmax - t.shape[1]
            t = F.pad(t, (0, 0, 0, pad, 0, 0))
        elif t.shape[1] > lmax:
            t = t[:, :lmax, :]
        out.append(t)
    return out, b0, lmax, d0


def _pack(tensor, base_dict):
    return ([[tensor, dict(base_dict)]],)


def _check_compat(tensors):
    """Проверить одинаковые B и dim (универсальность: одна модель/энкодер). Seq может отличаться."""
    if len(tensors) < 1:
        raise ValueError("нужен хотя бы один кондишен")
    b0, d0 = tensors[0].shape[0], tensors[0].shape[2]
    for i, t in enumerate(tensors):
        if t.shape[0] != b0 or t.shape[2] != d0:
            raise ValueError(
                f"несовместимые кондишены (разные модели?): вход {i+1} имеет {tuple(t.shape)}, "
                f"а вход 1 — {tuple(tensors[0].shape)}. "
                f"Смешивать можно только кондишены одной модели/энкодера."
            )
    return b0, d0


def _norm_tokens(t):
    return t / (t.norm(dim=-1, keepdim=True) + EPS)


def _set_and_batch(anchor, others, threshold):
    """anchor (La,D), others list of (Lk,D) — уже float.
    Возврат (Lout,D): общие вектора — токены anchor, похожие на ВСЕ остальные,
    усредненные со своими ближайшими соседями. Несовпавшие отбрасываются."""
    a_n = _norm_tokens(anchor)
    scores = torch.ones(anchor.shape[0], dtype=torch.float32)
    nearest = []
    for o in others:
        o_n = _norm_tokens(o)
        sim = a_n @ o_n.T  # (La, Lk)
        best_val, best_idx = sim.max(dim=1)
        scores = torch.minimum(scores, best_val)
        nearest.append(o[best_idx])  # (La, D)
    keep = scores >= threshold
    if int(keep.sum()) == 0:
        # fallback: лучшая single пара, чтобы не вернуть пустой кондишен
        i = int(scores.argmax())
        parts = [anchor[i:i+1]] + [n[i:i+1] for n in nearest]
        return torch.stack(parts, dim=0).mean(dim=0)
    picked = [anchor[keep]] + [n[keep] for n in nearest]
    return torch.stack(picked, dim=0).mean(dim=0)


def _pooled_max_sim(tokens, pools):
    """max косинусная схожесть каждого токена tokens со всеми токенами из pools."""
    t_n = _norm_tokens(tokens)
    best = torch.full((tokens.shape[0],), -1.0)
    for p in pools:
        sim = t_n @ _norm_tokens(p).T
        best = torch.maximum(best, sim.max(dim=1).values)
    return best


def set_op_and(tensors, threshold=0.7):
    _check_compat(tensors)
    outs = []
    for b in range(tensors[0].shape[0]):
        outs.append(_set_and_batch(tensors[0][b], [t[b] for t in tensors[1:]], threshold))
    lmax = max(o.shape[0] for o in outs)
    padded = []
    for o in outs:
        if o.shape[0] < lmax:
            pad = lmax - o.shape[0]
            o = F.pad(o, (0, 0, 0, pad, 0, 0))
        padded.append(o)
    return torch.stack(padded, dim=0)


def set_op_or(tensors):
    """Объединение: конкатенация всех векторов (дедупликация не нужна — сэмплер сам взвесит)."""
    _check_compat(tensors)
    return torch.cat(tensors, dim=1)


def set_op_xor(tensors, threshold=0.7):
    """Симметрическая разность: вектора, НЕ похожие ни на что в остальных входах."""
    _check_compat(tensors)
    b = tensors[0].shape[0]
    outs = []
    for bi in range(b):
        parts = []
        for m, t in enumerate(tensors):
            pools = [o[bi] for j, o in enumerate(tensors) if j != m]
            score = _pooled_max_sim(t[bi], pools)
            keep = score < threshold
            if int(keep.sum()) == 0:
                i = int(score.argmin())  # самый непохожий
                parts.append(t[bi][i:i+1])
            else:
                parts.append(t[bi][keep])
        outs.append(torch.cat(parts, dim=0))
    lmax = max(o.shape[0] for o in outs)
    padded = [o if o.shape[0] == lmax else F.pad(o, (0, 0, 0, lmax - o.shape[0], 0, 0)) for o in outs]
    return torch.stack(padded, dim=0)


def set_op_remove(tensors, threshold=0.7):
    """A NOT B1 NOT B2 ...: вектора A, НЕ похожие ни на что в остальных (чистка фона/света)."""
    _check_compat(tensors)
    b = tensors[0].shape[0]
    outs = []
    for bi in range(b):
        pools = [t[bi] for t in tensors[1:]]
        score = _pooled_max_sim(tensors[0][bi], pools)
        keep = score < threshold
        if int(keep.sum()) == 0:
            i = int(score.argmin())
            outs.append(tensors[0][bi][i:i+1])
        else:
            outs.append(tensors[0][bi][keep])
    lmax = max(o.shape[0] for o in outs)
    padded = [o if o.shape[0] == lmax else F.pad(o, (0, 0, 0, lmax - o.shape[0], 0, 0)) for o in outs]
    return torch.stack(padded, dim=0)


def op_add(tensors):
    aligned, _, _, _ = _align(tensors)
    return torch.stack(aligned, dim=0).sum(dim=0)


def op_subtract(tensors):
    aligned, _, _, _ = _align(tensors)
    r = aligned[0].clone()
    for t in aligned[1:]:
        r = r - t
    return r


def op_not(tensor):
    return -tensor


# ---------- общие миксины ----------

def _multi_input_types():
    opt = ContainsAnyDict()
    opt["cond_1"] = ("CONDITIONING", {"forceInput": True})
    opt["cond_2"] = ("CONDITIONING", {"forceInput": True})
    return {"required": {}, "optional": opt}


def _collect_tensors(fixed1, fixed2, kwargs):
    from_nodes = _collect_conds(COND_PREFIX, fixed1, fixed2, kwargs)
    if len(from_nodes) < 2:
        raise ValueError(f"подключите минимум 2 кондишена (сейчас {len(from_nodes)})")
    tensors, first_dict = [], {}
    for c in from_nodes:
        t, d = _single_tensor(c)
        tensors.append(t.float())
        if not first_dict:
            first_dict = d
    return tensors, first_dict


def _run_arith(fixed1, fixed2, kwargs, fn):
    tensors, first_dict = _collect_tensors(fixed1, fixed2, kwargs)
    return _pack(fn(tensors), first_dict)


def _run_set(fixed1, fixed2, kwargs, fn, threshold):
    tensors, first_dict = _collect_tensors(fixed1, fixed2, kwargs)
    return _pack(fn(tensors, threshold), first_dict)


def _threshold_input():
    opt = ContainsAnyDict()
    opt["cond_1"] = ("CONDITIONING", {"forceInput": True})
    opt["cond_2"] = ("CONDITIONING", {"forceInput": True})
    return {"required": {
        "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
    }, "optional": opt}


# ---------- ноды операций ----------

class CondAddZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return _multi_input_types()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, cond_1=None, cond_2=None, **kwargs):
        return _run_arith(cond_1, cond_2, kwargs, op_add)


class CondSubtractZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return _multi_input_types()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, cond_1=None, cond_2=None, **kwargs):
        return _run_arith(cond_1, cond_2, kwargs, op_subtract)


class CondAndZveroboy:
    """AND: общие вектора — токены cond_1, похожие на ВСЕ остальные, усредненные
    со своими ближайшими соседями. Несовпавшие отбрасываются."""
    @classmethod
    def INPUT_TYPES(cls):
        return _threshold_input()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, threshold=0.7, cond_1=None, cond_2=None, **kwargs):
        return _run_set(cond_1, cond_2, kwargs, set_op_and, threshold)


class CondOrZveroboy:
    """OR: объединение — конкатенация векторов всех входов."""
    @classmethod
    def INPUT_TYPES(cls):
        return _multi_input_types()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, cond_1=None, cond_2=None, **kwargs):
        tensors, first_dict = _collect_tensors(cond_1, cond_2, kwargs)
        return _pack(set_op_or(tensors), first_dict)


class CondXorZveroboy:
    """XOR: только различающиеся вектора (непохожие ни на что в остальных)."""
    @classmethod
    def INPUT_TYPES(cls):
        return _threshold_input()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, threshold=0.7, cond_1=None, cond_2=None, **kwargs):
        return _run_set(cond_1, cond_2, kwargs, set_op_xor, threshold)


class CondNandZveroboy:
    """NAND = NOT(AND)."""
    @classmethod
    def INPUT_TYPES(cls):
        return _threshold_input()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, threshold=0.7, cond_1=None, cond_2=None, **kwargs):
        tensors, first_dict = _collect_tensors(cond_1, cond_2, kwargs)
        return _pack(op_not(set_op_and(tensors, threshold)), first_dict)


class CondNorZveroboy:
    """NOR = NOT(OR)."""
    @classmethod
    def INPUT_TYPES(cls):
        return _multi_input_types()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, cond_1=None, cond_2=None, **kwargs):
        tensors, first_dict = _collect_tensors(cond_1, cond_2, kwargs)
        return _pack(op_not(set_op_or(tensors)), first_dict)


class CondXnorZveroboy:
    """XNOR = NOT(XOR)."""
    @classmethod
    def INPUT_TYPES(cls):
        return _threshold_input()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, threshold=0.7, cond_1=None, cond_2=None, **kwargs):
        tensors, first_dict = _collect_tensors(cond_1, cond_2, kwargs)
        return _pack(op_not(set_op_xor(tensors, threshold)), first_dict)


class CondNotZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, conditioning):
        t, d = _single_tensor(conditioning)
        return _pack(op_not(t.float()), d)


class CondRemoveZveroboy:
    """A NOT B1 NOT B2 ... — оставить в первом только вектора, НЕ похожие
    ни на что в остальных (чистка фона/света/локации)."""
    @classmethod
    def INPUT_TYPES(cls):
        return _threshold_input()
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "zveroboy/Condition"

    def run(self, threshold=0.7, cond_1=None, cond_2=None, **kwargs):
        return _run_set(cond_1, cond_2, kwargs, set_op_remove, threshold)


# ---------- save / load (совместимо с Comfyui-Condition-Utils: torch.save .ckpt) ----------

def _conditions_dir():
    if _HAS_FOLDER_PATHS:
        try:
            if "conditions" not in folder_paths.folder_names_and_paths:
                d = os.path.join(folder_paths.models_dir, "conditions")
                folder_paths.folder_names_and_paths["conditions"] = ([d], {".ckpt"})
            else:
                d = folder_paths.folder_names_and_paths["conditions"][0][0]
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            pass
    d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conditions")
    os.makedirs(d, exist_ok=True)
    return d


def _sanitize(name):
    base = os.path.basename(name.strip())
    if base.lower().endswith((".ckpt", ".pt", ".cond", ".safetensors")):
        stem = base[: base.rfind(".")]
        ext = base[base.rfind(".") :]
    else:
        stem, ext = base, ".ckpt"
    stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem) or "condition"
    return stem + ext


class CondSaveZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "filename": ("STRING", {"default": "condition", "multiline": False}),
        }}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "save"
    CATEGORY = "zveroboy/Condition"
    OUTPUT_NODE = True

    def save(self, conditioning, filename="condition"):
        save_dir = _conditions_dir()
        path = os.path.join(save_dir, _sanitize(filename))
        payload = []
        for entry in conditioning:
            t = entry[0].detach().cpu().clone() if torch.is_tensor(entry[0]) else entry[0]
            d = dict(entry[1]) if len(entry) > 1 and isinstance(entry[1], dict) else {}
            # тензоры внутри dict тоже на cpu для совместимости
            for k, v in list(d.items()):
                if torch.is_tensor(v):
                    d[k] = v.detach().cpu().clone()
            payload.append([t, d])
        torch.save(payload, path)
        print(f"[ConditionZveroboy] saved -> {path}")
        return (conditioning,)


class CondLoadZveroboy:
    @classmethod
    def INPUT_TYPES(cls):
        save_dir = _conditions_dir()
        try:
            files = [f for f in os.listdir(save_dir)
                     if f.lower().endswith((".ckpt", ".pt", ".cond"))]
        except Exception:
            files = []
        if _HAS_FOLDER_PATHS:
            try:
                extra = folder_paths.get_filename_list("conditions")
                for f in extra:
                    if f not in files:
                        files.append(f)
            except Exception:
                pass
        if not files:
            files = ["No files found"]
        return {"required": {"filename": (sorted(files),)}}
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "load"
    CATEGORY = "zveroboy/Condition"

    def load(self, filename):
        if filename == "No files found":
            raise FileNotFoundError("в папке conditions нет файлов")
        save_dir = _conditions_dir()
        path = os.path.join(save_dir, os.path.basename(filename))
        if not os.path.exists(path):
            # вдруг выбрано имя без расширения из folder_paths
            for ext in (".ckpt", ".pt", ".cond"):
                if os.path.exists(path + ext):
                    path = path + ext
                    break
        if not os.path.exists(path):
            raise FileNotFoundError(f"файл не найден: {path}")
        if path.lower().endswith(".safetensors"):
            from safetensors.torch import load_file as _load_sf
            sd = _load_sf(path, device="cpu")
            # ожидаем ключи cond / pooled_* от встроенного SaveConditioning — восстановим минимум
            if "cond" in sd:
                t = sd["cond"]
                d = {}
                for k, v in sd.items():
                    if k != "cond":
                        d[k] = v
                return ([[t, d]],)
            # иначе склеить всё как один тензор
            vals = list(sd.values())
            return ([[vals[0], {}]],)
        data = torch.load(path, map_location="cpu", weights_only=False)
        # формат Condition-Utils: list of [tensor, dict]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
            return (data,)
        # одиночный тензор — обернуть
        if torch.is_tensor(data):
            return ([[data, {}]],)
        raise ValueError(f"неизвестный формат файла: {path}")


NODE_CLASS_MAPPINGS = {
    "CondAddZveroboy": CondAddZveroboy,
    "CondSubtractZveroboy": CondSubtractZveroboy,
    "CondAndZveroboy": CondAndZveroboy,
    "CondOrZveroboy": CondOrZveroboy,
    "CondXorZveroboy": CondXorZveroboy,
    "CondNandZveroboy": CondNandZveroboy,
    "CondNorZveroboy": CondNorZveroboy,
    "CondXnorZveroboy": CondXnorZveroboy,
    "CondNotZveroboy": CondNotZveroboy,
    "CondRemoveZveroboy": CondRemoveZveroboy,
    "CondSaveZveroboy": CondSaveZveroboy,
    "CondLoadZveroboy": CondLoadZveroboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CondAddZveroboy": "Condition Add (Zveroboy)",
    "CondSubtractZveroboy": "Condition Subtract (Zveroboy)",
    "CondAndZveroboy": "Condition AND (Zveroboy)",
    "CondOrZveroboy": "Condition OR (Zveroboy)",
    "CondXorZveroboy": "Condition XOR (Zveroboy)",
    "CondNandZveroboy": "Condition NAND (Zveroboy)",
    "CondNorZveroboy": "Condition NOR (Zveroboy)",
    "CondXnorZveroboy": "Condition XNOR (Zveroboy)",
    "CondNotZveroboy": "Condition NOT (Zveroboy)",
    "CondRemoveZveroboy": "Condition Remove / A NOT B (Zveroboy)",
    "CondSaveZveroboy": "Condition Save (Zveroboy)",
    "CondLoadZveroboy": "Condition Load (Zveroboy)",
}
