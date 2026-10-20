"""Bounded-memory JSON-array parsing and source-bound scene annotation queries."""
import json
import os
import sqlite3
import uuid
from pathlib import Path

from trajectory_prediction.context.neighbors import NeighborIndex
from trajectory_prediction.utils.provenance import sha256_file


def stream_json_array(path, chunk_size=1 << 20):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk size must be a positive integer")
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer, eof, started, first, need_value = "", False, False, True, True
        while True:
            buffer = buffer.lstrip()
            if not buffer and not eof:
                buffer = stream.read(chunk_size)
                eof = not buffer
                continue
            if not started:
                if not buffer.startswith("["):
                    raise ValueError("metadata must be a JSON array")
                buffer, started = buffer[1:], True
                continue
            if buffer.startswith("]"):
                if need_value and not first:
                    raise ValueError("trailing comma in metadata")
                tail = buffer[1:]
                if tail.strip() or any(chunk.strip() for chunk in iter(lambda: stream.read(chunk_size), "")):
                    raise ValueError("unexpected trailing JSON content")
                return
            if not need_value:
                if not buffer.startswith(","):
                    raise ValueError("metadata array separator missing")
                buffer, need_value = buffer[1:], True
                continue
            try:
                value, end = decoder.raw_decode(buffer)
                if not eof and (end == len(buffer) or (isinstance(value, (int, float)) and buffer[end] not in ",] \t\r\n")):
                    more = stream.read(chunk_size)
                    eof = not more
                    buffer += more
                    continue
            except json.JSONDecodeError as exc:
                if eof or len(buffer) > 16 * 1024 * 1024:
                    raise ValueError("invalid or oversized metadata JSON record") from exc
                more = stream.read(chunk_size)
                eof = not more
                buffer += more
                continue
            yield value
            buffer, first, need_value = buffer[end:], False, False


class CompactMetadata:
    def __init__(self, dataroot, version, database):
        self.root, self.version, self.database = Path(dataroot), version, Path(database)
        self.tables, self.source_hashes = {}, {}
        for name in ("sample", "scene", "log", "category"):
            path = self.root / version / (name + ".json")
            self.tables[name] = {record["token"]: record for record in stream_json_array(path)}
        for name in ("sample", "scene", "log", "sample_annotation"):
            self.source_hashes[f"{version}/{name}.json"] = sha256_file(self.root / version / (name + ".json"))
        self.extra_hashes = {f"{version}/{name}.json": sha256_file(self.root / version / (name + ".json"))
                             for name in ("instance", "category")}
        self.categories = {r["token"]: self.tables["category"][r["category_token"]]["name"]
                           for r in stream_json_array(self.root / version / "instance.json")}
        self.scene_tokens = {r["name"]: token for token, r in self.tables["scene"].items()}
        self._ensure_database()

    def provenance(self):
        hashes = dict(self.source_hashes)
        for path in sorted((self.root / "maps" / "expansion").glob("*.json")):
            hashes[f"maps/expansion/{path.name}"] = sha256_file(path)
        if len(hashes) == len(self.source_hashes):
            raise ValueError("no expansion maps found")
        return hashes

    def _ensure_database(self):
        sources = {**self.source_hashes, **self.extra_hashes}
        manifest_path = self.database.with_suffix(".json")
        if self.database.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("source") != sources or manifest.get("schema_version") != 1:
                raise ValueError("compact metadata source mismatch")
            if sha256_file(self.database) != manifest.get("sha256"):
                raise ValueError("compact metadata checksum mismatch")
            return
        self.database.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.database.with_name(".building-" + uuid.uuid4().hex + ".sqlite")
        connection = sqlite3.connect(temporary)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE annotations(token TEXT PRIMARY KEY, sample TEXT, instance TEXT, scene TEXT, x REAL, y REAL, q0 REAL, q1 REAL, q2 REAL, q3 REAL)")
            batch, count = [], 0
            for record in stream_json_array(self.root / self.version / "sample_annotation.json"):
                sample = self.tables["sample"][record["sample_token"]]
                batch.append((record["token"], record["sample_token"], record["instance_token"],
                    sample["scene_token"], *record["translation"][:2], *record["rotation"]))
                if len(batch) >= 4096:
                    connection.executemany("INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
                    connection.commit()
                    count += len(batch)
                    batch.clear()
                    if count % 65536 == 0:
                        print(json.dumps({"metadata_index_records": count}), flush=True)
            if batch:
                connection.executemany("INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
            connection.execute("CREATE INDEX annotation_scene ON annotations(scene)")
            connection.commit()
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        os.replace(temporary, self.database)
        manifest_path.write_text(json.dumps({"schema_version": 1, "source": sources,
            "sha256": sha256_file(self.database)}, indent=2) + "\n", encoding="utf-8")

    def scene_index(self, scene, config=None):
        scene_token = self.scene_tokens.get(scene)
        samples = [r for r in self.tables["sample"].values() if r["scene_token"] == scene_token]
        connection = sqlite3.connect(self.database)
        try:
            connection.execute("PRAGMA query_only=ON")
            records = [{"token": token, "sample_token": sample, "instance_token": instance,
                        "translation": [x, y, 0.], "rotation": [q0, q1, q2, q3]}
                       for token, sample, instance, _, x, y, q0, q1, q2, q3 in connection.execute(
                           "SELECT * FROM annotations WHERE scene=?", (scene_token,))]
        finally:
            connection.close()
        categories = {r["instance_token"]: self.categories[r["instance_token"]] for r in records}
        index = NeighborIndex(samples, records, categories,
            {t: r["name"] for t, r in self.tables["scene"].items()}, config)
        index.extra_hashes = dict(self.extra_hashes)
        return index
