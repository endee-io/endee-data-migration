"""
core/checkpoint.py
──────────────────────────────────────────────────────────────────────────────
Persistence layer for migration progress.

The checkpoint is source/sink-agnostic.  It stores:
  • processed_count : total records successfully upserted across all runs
  • last_cursor     : opaque resume token produced by the source connector
                      (int for Milvus, scroll-UUID for Qdrant, etc.)
  • batch_number    : last batch that completed successfully
  • completed       : True once the source signals there is no more data

The pipeline reads the checkpoint at startup and writes it after every
successfully upserted batch.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import orjson

logger = logging.getLogger(__name__)

_PROCESSED_COUNT = "processed_count"
_LAST_CURSOR     = "last_offset"       # legacy name kept for file compatibility
_BATCH_NUMBER    = "batch_number"
_COMPLETED       = "completed"

_DEFAULTS: Dict[str, Any] = {
    _PROCESSED_COUNT: 0,
    _LAST_CURSOR:     None,
    _BATCH_NUMBER:    0,
    _COMPLETED:       False,
}


class MigrationCheckpoint:
    def __init__(self, checkpoint_file: str = "./migration_checkpoint.json"):
        self.checkpoint_file = checkpoint_file
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.checkpoint_file, "rb") as f:
                data = orjson.loads(f.read())
            logger.info(
                f"Loaded checkpoint: "
                f"{data.get(_PROCESSED_COUNT, 0)} records already processed"
            )
            return data
        except FileNotFoundError:
            logger.info("No checkpoint found — starting fresh migration")
            return dict(_DEFAULTS)
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}) — starting fresh")
            return dict(_DEFAULTS)

    def save(self) -> None:
        try:
            dirpath = os.path.dirname(self.checkpoint_file)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(self.checkpoint_file, "wb") as f:
                f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def update(self, batch_number: int, records_count: int, cursor: Any) -> None:
        """Call after each successfully upserted batch."""
        self.data[_PROCESSED_COUNT] += records_count
        self.data[_BATCH_NUMBER]    = batch_number
        self.data[_LAST_CURSOR]     = cursor
        self.save()

    def mark_completed(self) -> None:
        self.data[_COMPLETED] = True
        self.save()

    def clear(self) -> None:
        self.data = dict(_DEFAULTS)
        self.save()
        logger.info("Checkpoint cleared — fresh migration will start")

    def is_completed(self) -> bool:
        return bool(self.data.get(_COMPLETED, False))

    def get_processed_count(self) -> int:
        return int(self.data.get(_PROCESSED_COUNT, 0))

    def get_last_cursor(self) -> Any:
        return self.data.get(_LAST_CURSOR)

    # Alias used inside source connectors
    def get_last_offset(self) -> Any:
        return self.get_last_cursor()

    def get_batch_number(self) -> int:
        return int(self.data.get(_BATCH_NUMBER, 0))