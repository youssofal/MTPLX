"""CLI for MTPLX semantic memory and dreaming."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mtplx.dreaming import DreamingService
from mtplx.memory import MemoryStore, MemoryStoreError


def _store(args: argparse.Namespace) -> MemoryStore:
    return MemoryStore(getattr(args, "memory_dir", None))


def _emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif isinstance(payload, str):
        print(payload, end="" if payload.endswith("\n") else "\n")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _content(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).expanduser().read_text(encoding="utf-8")
    if args.content is not None:
        return args.content
    return sys.stdin.read()


def cmd_memory_public(args: argparse.Namespace) -> int:
    store = _store(args)
    action = args.memory_action
    try:
        if action == "init":
            _emit(store.initialize(), as_json=args.json)
            return 0
        if action == "status":
            _emit(store.status(), as_json=True)
            return 0
        if action == "list":
            documents = store.list_documents(scope=args.scope, limit=args.limit)
            payload = [doc.to_dict(include_content=False) for doc in documents]
            _emit(payload, as_json=args.json)
            return 0
        if action == "read":
            document = store.read(args.path)
            _emit(document.to_dict(include_content=not args.metadata_only), as_json=args.json)
            return 0
        if action == "search":
            hits = store.search(args.query, scope=args.scope, limit=args.limit)
            payload = [hit.to_dict() for hit in hits]
            _emit(payload, as_json=args.json)
            return 0
        if action == "context":
            context = store.build_context(
                args.query,
                scope=args.scope,
                limit=args.limit,
                max_chars=args.max_chars,
            )
            _emit(context.to_dict(), as_json=True)
            return 0
        if action == "history":
            _emit(store.history(args.path), as_json=args.json)
            return 0
        if action in {"write", "append"}:
            content = _content(args)
            if action == "append":
                document = store.append(
                    args.path,
                    content,
                    author=args.author,
                    session_id=args.session_id,
                    agent_id=args.agent_id,
                )
            else:
                document = store.write(
                    args.path,
                    content,
                    expected_sha256=args.expected_sha256,
                    author=args.author,
                    session_id=args.session_id,
                    agent_id=args.agent_id,
                )
            _emit(document.to_dict(include_content=False), as_json=args.json)
            return 0
        if action == "restore":
            document = store.restore(
                args.path,
                args.version,
                expected_sha256=args.expected_sha256,
                author=args.author,
                session_id=args.session_id,
                agent_id=args.agent_id,
            )
            _emit(document.to_dict(include_content=False), as_json=args.json)
            return 0
        if action == "transcript":
            messages = json.loads(args.messages)
            if not isinstance(messages, list):
                raise ValueError("--messages must be a JSON array")
            document = store.record_transcript(
                args.session_id,
                messages,
                agent_id=args.agent_id,
                request_id=args.request_id,
                model=args.model,
            )
            _emit(document.to_dict(include_content=False), as_json=args.json)
            return 0
        if action == "dream":
            service = DreamingService(store)
            try:
                result = service.run(max_transcripts=args.max_transcripts)
            finally:
                service.close()
            if args.apply:
                service = DreamingService(store)
                try:
                    result = service.apply(result["run_id"])
                finally:
                    service.close()
            _emit(result, as_json=True)
            return 0
        if action == "apply":
            service = DreamingService(store)
            try:
                _emit(service.apply(args.run_id), as_json=True)
            finally:
                service.close()
            return 0
        raise ValueError(f"unknown memory action: {action}")
    except (MemoryStoreError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {"error": type(exc).__name__, "detail": str(exc)}
        _emit(payload, as_json=True)
        return 2


__all__ = ["cmd_memory_public"]
