"""将已冻结的位置实验协议与代码复制到研究仓库。"""
from pathlib import Path
from shutil import copy2

repo = Path(__file__).resolve().parents[1]
source = repo.parent / "deliverables" / "single_full_mcs_mask_multiseed_20261031" / "coset_position_20261031"
target = repo / "experiments" / "20261101_coset_position_m8"

items = [
    (source / "实验合同与自检_20261101.md", target / "protocol" / "实验合同与自检_20261101.md"),
    (source / "coset_preflight.json", target / "protocol" / "coset_preflight.json"),
    (source.parent / "coset_preflight.py", target / "code" / "coset_preflight.py"),
]
items += [(p, target / "protocol" / "mapping_gates" / p.name) for p in sorted((source / "mapping_gates").glob("*.json"))]
items += [(p, target / "code" / p.name) for p in sorted((source / "code").glob("*.py")) + sorted((source / "code").glob("*.sh"))]
for src, dst in items:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy2(src, dst)
    print(dst.relative_to(repo))
