"""生成 v2 正式候选数据或极小 QA-S 数据。

默认只允许每个 split 不超过 32 个源；正式批量必须显式提供
``--confirm-formal-bulk``，且仍需研究者另行批准后才能运行。
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import v2_formal_core as core


TASKS = ("single", "dual")
STRING = h5py.string_dtype("utf-8")


def _generate_row_worker(arguments):
    """进程池工作单元；随机性仅由命名seed决定，与worker数量无关。"""
    task, split, row, master_seed, mods, slots, esn0_grid = arguments
    source = core.generate_clean_source(task, split, row, master_seed, mods, slots)
    views = [core.generate_view(source, task, split, master_seed, float(value)) for value in esn0_grid]
    return source, views


def _pad(values: np.ndarray | list[Any], width: int, fill: Any, dtype: Any) -> np.ndarray:
    result = np.full(width, fill, dtype=dtype)
    result[: len(values)] = values
    return result


def _create_source_datasets(h5: h5py.File, count: int, store_components: bool) -> None:
    h5.create_dataset("clean_full_iq", (count, core.N_FULL, 2), dtype="f4", chunks=(1, core.N_FULL, 2), compression="lzf")
    if store_components:
        h5.create_dataset("component_full_iq", (count, 2, core.N_FULL, 2), dtype="f4", chunks=(1, 1, core.N_FULL, 2), compression="lzf", fillvalue=np.nan)
    h5.create_dataset("source_uid", (count,), dtype=STRING)
    h5.create_dataset("component_uid", (count, 2), dtype=STRING)
    h5.create_dataset("n_signal", (count,), dtype="u1")
    h5.create_dataset("occupied_slots", (count, 2), dtype="i2", fillvalue=-1)
    h5.create_dataset("modulation_labels", (count, 2), dtype="i1", fillvalue=-1)
    h5.create_dataset("y_occ", (count, 24), dtype="u1")
    h5.create_dataset("y_mod", (count, 24), dtype="i1")
    h5.create_dataset("phase_rad", (count, 2), dtype="f8", fillvalue=np.nan)
    h5.create_dataset("symbol_seed", (count, 2), dtype="u8")
    h5.create_dataset("phase_seed", (count, 2), dtype="u8")
    h5.create_dataset("component_power_full", (count, 2), dtype="f8", fillvalue=np.nan)
    h5.create_dataset("clean_full_hash", (count,), dtype=STRING)
    h5.create_dataset("component_hash", (count, 2), dtype=STRING)
    h5.create_dataset("symbol_hash", (count, 2), dtype=STRING)


def _create_view_datasets(h5: h5py.File, count: int, n_snr: int, store_noisy_full: bool) -> None:
    h5.create_dataset("mcs_iq", (count, n_snr, core.N_MCS, 2 * core.M), dtype="f4", chunks=(1, 1, core.N_MCS, 2 * core.M), compression="lzf")
    if store_noisy_full:
        h5.create_dataset("noisy_full_iq", (count, n_snr, core.N_FULL, 2), dtype="f4", chunks=(1, 1, core.N_FULL, 2), compression="lzf")
    h5.create_dataset("view_uid", (count, n_snr), dtype=STRING)
    h5.create_dataset("noise_seed", (count, n_snr), dtype="u8")
    for name in (
        "target_esn0_db", "target_full_component_snr_db", "target_slot50_snr_db",
        "target_total_snr_db", "measured_full_total_snr_db",
    ):
        h5.create_dataset(name, (count, n_snr), dtype="f4")
    for name in ("target_noise_power_full", "noise_power_full_measured"):
        h5.create_dataset(name, (count, n_snr), dtype="f8")
    h5.create_dataset("measured_full_component_snr_db", (count, n_snr, 2), dtype="f4", fillvalue=np.nan)
    h5.create_dataset("mcs_noise_power_per_branch", (count, n_snr, core.M), dtype="f8")
    h5.create_dataset("mcs_clean_power_per_branch", (count, n_snr, core.M), dtype="f8")
    h5.create_dataset("mcs_snr_per_branch_db", (count, n_snr, core.M), dtype="f4")


def _write_source(h5: h5py.File, row: int, source: dict[str, Any], store_components: bool) -> None:
    n_signal = int(source["n_signal"])
    clean = source["clean_full"]
    h5["clean_full_iq"][row, :, 0] = clean.real
    h5["clean_full_iq"][row, :, 1] = clean.imag
    if store_components:
        for component in range(n_signal):
            value = source["components"][component]
            h5["component_full_iq"][row, component, :, 0] = value.real
            h5["component_full_iq"][row, component, :, 1] = value.imag
    h5["source_uid"][row] = source["source_uid"]
    h5["component_uid"][row] = _pad(source["component_uid"], 2, "", object)
    h5["n_signal"][row] = n_signal
    h5["occupied_slots"][row] = _pad(source["occupied_slots"], 2, -1, np.int16)
    h5["modulation_labels"][row] = _pad(source["modulation_labels"], 2, -1, np.int8)
    h5["y_occ"][row] = source["y_occ"]
    h5["y_mod"][row] = source["y_mod"]
    h5["phase_rad"][row] = _pad(source["phases_rad"], 2, np.nan, np.float64)
    h5["symbol_seed"][row] = _pad(source["symbol_seed"], 2, 0, np.uint64)
    h5["phase_seed"][row] = _pad(source["phase_seed"], 2, 0, np.uint64)
    h5["component_power_full"][row] = _pad(source["component_power_full"], 2, np.nan, np.float64)
    h5["clean_full_hash"][row] = source["clean_full_hash"]
    h5["component_hash"][row] = _pad(source["component_hash"], 2, "", object)
    h5["symbol_hash"][row] = _pad(source["symbol_hash"], 2, "", object)


def _write_view(h5: h5py.File, row: int, snr_index: int, view: dict[str, Any], n_signal: int, store_noisy_full: bool) -> None:
    h5["mcs_iq"][row, snr_index] = view["mcs_iq"]
    if store_noisy_full:
        h5["noisy_full_iq"][row, snr_index, :, 0] = view["noisy_full"].real
        h5["noisy_full_iq"][row, snr_index, :, 1] = view["noisy_full"].imag
    for name in ("view_uid", "noise_seed", "target_esn0_db", "target_full_component_snr_db", "target_slot50_snr_db", "target_total_snr_db", "target_noise_power_full", "noise_power_full_measured", "measured_full_total_snr_db"):
        h5[name][row, snr_index] = view[name]
    h5["measured_full_component_snr_db"][row, snr_index] = _pad(view["measured_full_component_snr_db"], 2, np.nan, np.float32)
    h5["mcs_noise_power_per_branch"][row, snr_index] = view["mcs_noise_power_per_branch"]
    h5["mcs_clean_power_per_branch"][row, snr_index] = view["mcs_clean_power_per_branch"]
    h5["mcs_snr_per_branch_db"][row, snr_index] = view["mcs_snr_per_branch_db"]


def write_task_split(
    output: Path,
    task: str,
    split: str,
    count: int,
    master_seed: int,
    esn0_grid: np.ndarray,
    config_hash: str,
    generator_hash: str,
    store_noisy_full: bool,
    store_components: bool,
    workers: int,
) -> dict[str, Any]:
    folder = output / task / split
    folder.mkdir(parents=True, exist_ok=False)
    schedule = core.split_schedule(task, split, count, master_seed)
    manifest_rows: list[dict[str, Any]] = []
    h5_path = folder / "dataset.h5"
    with h5py.File(h5_path, "w") as h5:
        _create_source_datasets(h5, count, store_components)
        _create_view_datasets(h5, count, len(esn0_grid), store_noisy_full)
        h5.attrs["generator_version"] = core.GENERATOR_VERSION
        h5.attrs["generator_sha256"] = generator_hash
        h5.attrs["config_sha256"] = config_hash
        h5.attrs["task"] = task
        h5.attrs["split"] = split
        h5.attrs["input_dtype"] = "float32"
        h5.attrs["metadata_forbidden_as_model_input"] = True
        h5.attrs["full_mcs_pairing"] = "mcs[n,m] = noisy_full[40*n + Coset[m]]"
        h5.attrs["Coset"] = core.COSET
        h5.attrs["esn0_grid_db"] = esn0_grid
        worker_arguments = (
            (task, split, row, master_seed, schedule["mods_component_order"][row],
             schedule["slots_component_order"][row], esn0_grid)
            for row in range(count)
        )
        executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        iterator = executor.map(_generate_row_worker, worker_arguments, chunksize=1) if executor else map(_generate_row_worker, worker_arguments)
        for row, (source, views) in enumerate(iterator):
            _write_source(h5, row, source, store_components)
            view_rows = []
            for snr_index, (esn0_db, view) in enumerate(zip(esn0_grid, views)):
                _write_view(h5, row, snr_index, view, int(source["n_signal"]), store_noisy_full)
                view_rows.append({
                    "view_uid": view["view_uid"],
                    "noise_seed": int(view["noise_seed"]),
                    "target_esn0_db": float(view["target_esn0_db"]),
                    "target_full_component_snr_db": float(view["target_full_component_snr_db"]),
                })
            manifest_rows.append({
                "row": row,
                "source_uid": source["source_uid"],
                "component_uid": source["component_uid"],
                "n_signal": int(source["n_signal"]),
                "occupied_slots": source["occupied_slots"].tolist(),
                "modulation_labels": source["modulation_labels"].tolist(),
                "phase_rad": source["phases_rad"].tolist(),
                "symbol_seed": [int(x) for x in source["symbol_seed"]],
                "phase_seed": [int(x) for x in source["phase_seed"]],
                "clean_full_hash": source["clean_full_hash"],
                "component_hash": source["component_hash"],
                "symbol_hash": source["symbol_hash"],
                "views": view_rows,
            })
        if executor is not None:
            executor.shutdown(wait=True)
    manifest_path = folder / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
    return {
        "task": task,
        "split": split,
        "sources": count,
        "h5": str(h5_path.relative_to(output)),
        "manifest": str(manifest_path.relative_to(output)),
        "manifest_sha256": core.sha256_file(manifest_path),
        "h5_sha256": core.sha256_file(h5_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 正式候选数据生成器")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS), help="仅生成获批任务")
    parser.add_argument("--seed", type=int, default=261028)
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--final-test", "--test-count", dest="test_count", type=int, default=16)
    parser.add_argument("--single-train", type=int)
    parser.add_argument("--single-validation", type=int)
    parser.add_argument("--single-test-count", type=int)
    parser.add_argument("--dual-train", type=int)
    parser.add_argument("--dual-validation", type=int)
    parser.add_argument("--dual-test-count", type=int)
    parser.add_argument("--test-role", choices=("final_test", "development_test"), default="final_test")
    snr_group = parser.add_mutually_exclusive_group()
    snr_group.add_argument("--esn0-grid", type=float, nargs="+", help="符号能量口径；与模型入口 SNR 相差20 dB")
    snr_group.add_argument(
        "--mcs-snr-grid", type=float, nargs="+",
        help="模型入口处每分量 MCS 采样点 SNR；推荐用于训练/测试网格",
    )
    parser.add_argument("--store-noisy-full", action="store_true")
    parser.add_argument("--store-components", action="store_true")
    parser.add_argument("--confirm-formal-bulk", action="store_true", help="仅在研究者另行批准正式批量生成后使用")
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)), help="并行生成波形，H5仍由主进程顺序写入")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ("train", "validation", args.test_role)
    shared_counts = {"train": args.train, "validation": args.validation, args.test_role: args.test_count}
    counts_by_task = {
        "single": {
            "train": args.single_train if args.single_train is not None else shared_counts["train"],
            "validation": args.single_validation if args.single_validation is not None else shared_counts["validation"],
            args.test_role: args.single_test_count if args.single_test_count is not None else shared_counts[args.test_role],
        },
        "dual": {
            "train": args.dual_train if args.dual_train is not None else shared_counts["train"],
            "validation": args.dual_validation if args.dual_validation is not None else shared_counts["validation"],
            args.test_role: args.dual_test_count if args.dual_test_count is not None else shared_counts[args.test_role],
        },
    }
    if any(value < 1 for counts in counts_by_task.values() for value in counts.values()):
        raise SystemExit("所有 split 的源数量必须为正整数")
    selected_tasks = tuple(args.tasks)
    if max(value for task in selected_tasks for value in counts_by_task[task].values()) > 32 and not args.confirm_formal_bulk:
        raise SystemExit("P0 门控：每 split 超过32源，必须在获得正式批量授权后显式提供 --confirm-formal-bulk")
    if args.mcs_snr_grid is not None:
        requested_mcs_snr_grid = np.asarray(args.mcs_snr_grid, dtype=np.float64)
        esn0_grid = core.mcs_component_snr_to_esn0_db(requested_mcs_snr_grid)
        requested_snr_axis = "mcs_component_snr_db"
    else:
        esn0_values = args.esn0_grid if args.esn0_grid is not None else core.DEFAULT_DEVELOPMENT_ESN0_GRID.tolist()
        esn0_grid = np.asarray(esn0_values, dtype=np.float64)
        requested_mcs_snr_grid = core.esn0_to_mcs_component_snr_db(esn0_grid)
        requested_snr_axis = "esn0_db"
    if len(esn0_grid) == 0 or not np.all(np.isfinite(esn0_grid)) or len(np.unique(esn0_grid)) != len(esn0_grid):
        raise SystemExit("Es/N0 网格必须为有限且互异的数值")
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖已有目录: {args.output}")
    if args.workers < 1:
        raise SystemExit("workers 必须为正整数")
    core.validate_frozen_physics()
    generator_path = Path(__file__).resolve()
    code_files = [generator_path, Path(core.__file__).resolve()]
    code_hashes = {path.name: core.sha256_file(path) for path in code_files}
    config = {
        "generator_version": core.GENERATOR_VERSION,
        "master_seed": args.seed,
        "counts_by_task": counts_by_task,
        "tasks": list(selected_tasks),
        "splits": list(splits),
        "requested_snr_axis": requested_snr_axis,
        "mcs_component_snr_grid_db": requested_mcs_snr_grid.tolist(),
        "esn0_grid_db": esn0_grid.tolist(),
        "full_component_snr_grid_db": core.esn0_to_full_component_snr_db(esn0_grid).tolist(),
        "snr_grid_status": "DEVELOPMENT_SCAN_NOT_FORMALLY_FROZEN",
        "store_noisy_full": args.store_noisy_full,
        "store_components": args.store_components,
        "execution_workers": args.workers,
        "physical": core.frozen_physical_config(),
        "label_protocol": "single_13_balanced; dual_169_ordered_balanced; independent_layout_without_replacement; frequency_sorted",
        "split_protocol": "source_uid_strict_isolation_train_validation_final_test",
        "code_sha256": code_hashes,
    }
    config_hash = core.canonical_hash(config)
    config["config_sha256"] = config_hash
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    entries = []
    generator_hash = core.canonical_hash(code_hashes)
    for task in selected_tasks:
        for split in splits:
            entries.append(write_task_split(
                args.output, task, split, counts_by_task[task][split], args.seed, esn0_grid,
                config_hash, generator_hash, args.store_noisy_full, args.store_components, args.workers,
            ))
    root_manifest = {
        "status": "GENERATED_AWAITING_QA",
        "role": "QA-S" if max(value for task in selected_tasks for value in counts_by_task[task].values()) <= 32 else "DEVELOPMENT_OR_FORMAL_CANDIDATE",
        "forbidden_as_final_evidence_until_qa_f_pass": True,
        "config_sha256": config_hash,
        "generator_sha256": generator_hash,
        "entries": entries,
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": root_manifest["status"], "output": str(args.output.resolve()), "config_sha256": config_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
