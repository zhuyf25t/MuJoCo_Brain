"""SQLite checkpoints and image archives shared by the task-stage graph."""

import base64
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from .llm_common import encode_image_block


class EpisodeMemory:
    def __init__(self, db_path, trace_root):
        self.db_path, self.trace_root = Path(db_path), Path(trace_root)
        self._db = self._images = self._lock = None

    def open(self):
        if self._db is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(str(self.db_path) + ".lock", "a+b")
        try:
            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise RuntimeError("此 LangGraph 数据库已有运行实例，请关闭它或指定独立 db_path") from None
        self._lock = lock
        try:
            self._db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            self.checkpointer = SqliteSaver(self._db)
            self.checkpointer.setup()
            # Checkpoint writes can run on another thread; image IO uses its own connection.
            self._images = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            self._images.execute("CREATE TABLE IF NOT EXISTS brain_images (id TEXT PRIMARY KEY, jpeg BLOB NOT NULL)")
            self._images.commit()
        except BaseException:
            self.close()
            raise

    def new_episode(self, thread_id):
        self.checkpointer.delete_thread(thread_id)
        self._images.execute("DELETE FROM brain_images")
        self._images.commit()
        self.trace_dir = self.trace_root / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
        (self.trace_dir / "images").mkdir(parents=True, exist_ok=True)

    def save_image(self, rgb, history_n):
        encoded = encode_image_block(rgb, fmt="openai")["image_url"]["url"]
        jpeg = base64.b64decode(encoded.split(",", 1)[1])
        image_id = hashlib.sha256(jpeg).hexdigest()
        self._images.execute("INSERT OR IGNORE INTO brain_images VALUES (?, ?)", (image_id, jpeg))
        self._images.commit()
        path = self.trace_dir / "images" / f"{image_id}.jpg"
        if not path.exists():
            path.write_bytes(jpeg)
        return {"image_id": image_id, "history_n": history_n}

    def image_block(self, frame):
        row = self._images.execute("SELECT jpeg FROM brain_images WHERE id=?", (frame["image_id"],)).fetchone()
        if row is None:
            raise RuntimeError("参考图片不存在")
        return {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(row[0]).decode()}}

    @staticmethod
    def archive_messages(messages, images):
        archived = deepcopy(messages)
        frames = iter(images)
        for part in archived[1]["content"]:
            if part["type"] == "image_url":
                part["image_url"]["url"] = "images/" + next(frames)[1]["image_id"] + ".jpg"
        return archived

    def log(self, record):
        with (self.trace_dir / "rounds.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        for connection in (self._images, self._db):
            if connection is not None:
                connection.close()
        self._images = self._db = None
        if self._lock:
            self._lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None
