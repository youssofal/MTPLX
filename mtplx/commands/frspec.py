"""``mtplx frspec``: FR-Spec draft vocabulary tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..frspec_build import (
    DEFAULT_ROLES,
    PRIVACY_NOTICE,
    BuildRequest,
    FrspecBuildError,
    build_frspec_table,
    format_report,
    format_tokenizer_groups,
    group_packs_by_tokenizer,
    read_exclude_file,
)


def cmd_frspec(args: argparse.Namespace) -> int:
    if getattr(args, "frspec_action", None) == "build":
        return _cmd_build(args)
    print("usage: mtplx frspec build --model <model> --input <path> --out <table.npy>", file=sys.stderr)
    return 2


def _cmd_build(args: argparse.Namespace) -> int:
    if args.list_models:
        print(_local_pack_listing(args))
        return 0
    missing = [flag for flag, value in (("--input", args.input), ("--out", args.out)) if not value]
    if missing:
        print(f"mtplx frspec build: missing {' and '.join(missing)}", file=sys.stderr)
        return 2
    try:
        tokenizer_json = _tokenizer_json(args)
    except FrspecBuildError as exc:
        print(f"mtplx frspec build: {exc}\n", file=sys.stderr)
        print(_local_pack_listing(args), file=sys.stderr)
        return 2
    try:
        request = BuildRequest(
            tokenizer_json=tokenizer_json,
            inputs=list(args.input),
            out=Path(args.out).expanduser(),
            rows=int(args.rows),
            fill_from=args.fill_from,
            order=args.order,
            holdout=float(args.holdout),
            max_file_bytes=int(args.max_file_bytes),
            roles=tuple(args.roles.split(",")) if args.roles else DEFAULT_ROLES,
            exclude_strings=read_exclude_file(args.exclude_tokens) if args.exclude_tokens else [],
        )
        report = build_frspec_table(request)
    except (FrspecBuildError, OSError) as exc:
        print(f"mtplx frspec build: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({**report, "notice": PRIVACY_NOTICE}, indent=2))
    else:
        print(format_report(report))
    return 0


def _tokenizer_json(args: argparse.Namespace) -> Path:
    """``tokenizer.json`` for ``--model``, resolved the way ``serve`` resolves it.

    A local directory or repo id goes through ``hf_loader.resolve_model_path``
    (MTPLX library roots, ``--cache-dir``, ``--model-search-dir``, then the
    Hugging Face cache); a name as ``mtplx models`` prints it is matched
    against that listing as well.
    """

    from ..hf_loader import resolve_model_path

    model = args.model
    if not model:
        raise FrspecBuildError("--model is required")
    local = Path(model).expanduser()
    if local.is_file() and local.name == "tokenizer.json":
        return local
    try:
        path = resolve_model_path(model, cache_dir=args.cache_dir, search_dirs=args.model_search_dirs)
    except (FileNotFoundError, ValueError) as exc:
        path = _listed_pack(args, model)
        if path is None:
            raise FrspecBuildError(str(exc)) from exc
    tokenizer_json = Path(path) / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FrspecBuildError(f"no tokenizer.json in {path}")
    return tokenizer_json


def _local_packs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    from ..hf_loader import list_cached_models

    rows = list_cached_models(cache_dir=args.cache_dir, search_dirs=args.model_search_dirs)
    return [(row.repo_id, row.path) for row in rows]


def _listed_pack(args: argparse.Namespace, model: str) -> Path | None:
    wanted = model.strip().lower()
    for name, path in _local_packs(args):
        if wanted in {name.lower(), path.name.lower()}:
            return path
    return None


def _local_pack_listing(args: argparse.Namespace) -> str:
    groups, unusable = group_packs_by_tokenizer(_local_packs(args))
    return format_tokenizer_groups(groups, unusable)
