#!/usr/bin/env python3
"""
hecbench_results.py - Result collection and storage framework for HeCBench

This module provides:
- SQLite database for storing benchmark results
- Export to JSON/CSV formats
- Query and analysis utilities
- Historical comparison

Usage:
    from hecbench_results import ResultsDB

    db = ResultsDB("results.db")
    db.store_result(benchmark="jacobi", model="cuda", value=1.234, ...)
    results = db.query(benchmark="jacobi")
    db.export_csv("results.csv")
"""

import csv
import json
import os
import sqlite3
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class BenchmarkResult:
    """A single benchmark result."""
    benchmark: str
    model: str
    value: float
    unit: str = "ms"
    timestamp: str = ""
    hostname: str = ""
    gpu_name: str = ""
    gpu_arch: str = ""
    compiler: str = ""
    commit_hash: str = ""
    wall_time: float = 0.0
    iterations: int = 1
    args: str = ""
    success: bool = True
    error_message: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.hostname:
            import socket
            self.hostname = socket.gethostname()


@dataclass
class BenchmarkStats:
    """Statistics for a set of benchmark results."""
    benchmark: str
    model: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    unit: str = "ms"


class ResultsDB:
    """SQLite database for storing and querying benchmark results."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        benchmark TEXT NOT NULL,
        model TEXT NOT NULL,
        value REAL NOT NULL,
        unit TEXT DEFAULT 'ms',
        timestamp TEXT NOT NULL,
        hostname TEXT,
        gpu_name TEXT,
        gpu_arch TEXT,
        compiler TEXT,
        commit_hash TEXT,
        wall_time REAL,
        iterations INTEGER DEFAULT 1,
        args TEXT,
        success INTEGER DEFAULT 1,
        error_message TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_benchmark ON results(benchmark);
    CREATE INDEX IF NOT EXISTS idx_model ON results(model);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON results(timestamp);
    CREATE INDEX IF NOT EXISTS idx_benchmark_model ON results(benchmark, model);

    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        hostname TEXT,
        gpu_name TEXT,
        gpu_arch TEXT,
        compiler TEXT,
        commit_hash TEXT,
        config TEXT
    );

    CREATE TABLE IF NOT EXISTS run_results (
        run_id INTEGER,
        result_id INTEGER,
        FOREIGN KEY (run_id) REFERENCES runs(id),
        FOREIGN KEY (result_id) REFERENCES results(id)
    );
    """

    def __init__(self, db_path: str = "hecbench_results.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # --- Result Storage ---

    def store_result(self, result: BenchmarkResult) -> int:
        """Store a single benchmark result. Returns the result ID."""
        cursor = self.conn.execute("""
            INSERT INTO results (
                benchmark, model, value, unit, timestamp, hostname,
                gpu_name, gpu_arch, compiler, commit_hash, wall_time,
                iterations, args, success, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.benchmark, result.model, result.value, result.unit,
            result.timestamp, result.hostname, result.gpu_name,
            result.gpu_arch, result.compiler, result.commit_hash,
            result.wall_time, result.iterations, result.args,
            1 if result.success else 0, result.error_message
        ))
        self.conn.commit()
        return cursor.lastrowid

    def store_results(self, results: List[BenchmarkResult]) -> List[int]:
        """Store multiple benchmark results. Returns list of result IDs."""
        ids = []
        for result in results:
            ids.append(self.store_result(result))
        return ids

    # --- Run Management ---

    def start_run(self, name: str = None, description: str = None,
                  config: Dict = None) -> int:
        """Start a new benchmark run. Returns the run ID."""
        import socket
        hostname = socket.gethostname()
        started_at = datetime.now().isoformat()
        config_json = json.dumps(config) if config else None

        cursor = self.conn.execute("""
            INSERT INTO runs (name, description, started_at, hostname, config)
            VALUES (?, ?, ?, ?, ?)
        """, (name, description, started_at, hostname, config_json))
        self.conn.commit()
        return cursor.lastrowid

    def finish_run(self, run_id: int, gpu_name: str = None,
                   gpu_arch: str = None, compiler: str = None,
                   commit_hash: str = None):
        """Mark a run as finished."""
        finished_at = datetime.now().isoformat()
        self.conn.execute("""
            UPDATE runs SET
                finished_at = ?,
                gpu_name = COALESCE(?, gpu_name),
                gpu_arch = COALESCE(?, gpu_arch),
                compiler = COALESCE(?, compiler),
                commit_hash = COALESCE(?, commit_hash)
            WHERE id = ?
        """, (finished_at, gpu_name, gpu_arch, compiler, commit_hash, run_id))
        self.conn.commit()

    def add_result_to_run(self, run_id: int, result_id: int):
        """Associate a result with a run."""
        self.conn.execute("""
            INSERT INTO run_results (run_id, result_id) VALUES (?, ?)
        """, (run_id, result_id))
        self.conn.commit()

    # --- Querying ---

    def query(self, benchmark: str = None, model: str = None,
              since: str = None, until: str = None,
              hostname: str = None, limit: int = None) -> List[Dict]:
        """Query results with optional filters."""
        conditions = []
        params = []

        if benchmark:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)
        if hostname:
            conditions.append("hostname = ?")
            params.append(hostname)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""

        cursor = self.conn.execute(f"""
            SELECT * FROM results
            WHERE {where_clause}
            ORDER BY timestamp DESC
            {limit_clause}
        """, params)

        return [dict(row) for row in cursor.fetchall()]

    def get_latest(self, benchmark: str, model: str) -> Optional[Dict]:
        """Get the most recent result for a benchmark/model combination."""
        results = self.query(benchmark=benchmark, model=model, limit=1)
        return results[0] if results else None

    def get_stats(self, benchmark: str = None, model: str = None,
                  since: str = None) -> List[BenchmarkStats]:
        """Get statistics for benchmark results."""
        conditions = ["success = 1"]
        params = []

        if benchmark:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        if model:
            conditions.append("model = ?")
            params.append(model)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)

        where_clause = " AND ".join(conditions)

        cursor = self.conn.execute(f"""
            SELECT
                benchmark, model, unit,
                COUNT(*) as count,
                AVG(value) as mean,
                MIN(value) as min_val,
                MAX(value) as max_val,
                GROUP_CONCAT(value) as values
            FROM results
            WHERE {where_clause}
            GROUP BY benchmark, model
            ORDER BY benchmark, model
        """, params)

        stats = []
        for row in cursor.fetchall():
            values = [float(v) for v in row['values'].split(',')]
            median = statistics.median(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

            stats.append(BenchmarkStats(
                benchmark=row['benchmark'],
                model=row['model'],
                count=row['count'],
                mean=row['mean'],
                median=median,
                std_dev=std_dev,
                min_val=row['min_val'],
                max_val=row['max_val'],
                unit=row['unit']
            ))

        return stats

    def get_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent benchmark runs."""
        cursor = self.conn.execute("""
            SELECT r.*, COUNT(rr.result_id) as result_count
            FROM runs r
            LEFT JOIN run_results rr ON r.id = rr.run_id
            GROUP BY r.id
            ORDER BY r.started_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_run_results(self, run_id: int) -> List[Dict]:
        """Get all results for a specific run."""
        cursor = self.conn.execute("""
            SELECT res.* FROM results res
            JOIN run_results rr ON res.id = rr.result_id
            WHERE rr.run_id = ?
            ORDER BY res.benchmark, res.model
        """, (run_id,))
        return [dict(row) for row in cursor.fetchall()]

    # --- Comparison ---

    def compare(self, benchmark: str = None, model: str = None,
                run1_id: int = None, run2_id: int = None) -> List[Dict]:
        """Compare results between two runs or time periods."""
        if run1_id and run2_id:
            # Compare specific runs
            results1 = {(r['benchmark'], r['model']): r
                       for r in self.get_run_results(run1_id)}
            results2 = {(r['benchmark'], r['model']): r
                       for r in self.get_run_results(run2_id)}
        else:
            # Compare latest vs previous
            raise NotImplementedError("Time-based comparison not yet implemented")

        comparisons = []
        all_keys = set(results1.keys()) | set(results2.keys())

        for key in sorted(all_keys):
            bench, mod = key
            if benchmark and bench != benchmark:
                continue
            if model and mod != model:
                continue

            r1 = results1.get(key)
            r2 = results2.get(key)

            if r1 and r2:
                diff = r2['value'] - r1['value']
                diff_pct = (diff / r1['value'] * 100) if r1['value'] != 0 else 0

                comparisons.append({
                    'benchmark': bench,
                    'model': mod,
                    'value1': r1['value'],
                    'value2': r2['value'],
                    'diff': diff,
                    'diff_pct': diff_pct,
                    'unit': r1.get('unit', 'ms')
                })

        return comparisons

    # --- Export ---

    def export_json(self, output_path: str, benchmark: str = None,
                    model: str = None, since: str = None):
        """Export results to JSON file."""
        results = self.query(benchmark=benchmark, model=model, since=since)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        return len(results)

    def export_csv(self, output_path: str, benchmark: str = None,
                   model: str = None, since: str = None):
        """Export results to CSV file."""
        results = self.query(benchmark=benchmark, model=model, since=since)

        if not results:
            return 0

        fieldnames = list(results[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        return len(results)

    def export_stats_csv(self, output_path: str, benchmark: str = None,
                         model: str = None, since: str = None):
        """Export statistics to CSV file."""
        stats = self.get_stats(benchmark=benchmark, model=model, since=since)

        if not stats:
            return 0

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['benchmark', 'model', 'count', 'mean', 'median',
                           'std_dev', 'min', 'max', 'unit'])
            for s in stats:
                writer.writerow([
                    s.benchmark, s.model, s.count, s.mean, s.median,
                    s.std_dev, s.min_val, s.max_val, s.unit
                ])

        return len(stats)

    # --- Import ---

    def import_json(self, input_path: str) -> int:
        """Import results from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        count = 0
        for item in data:
            result = BenchmarkResult(
                benchmark=item['benchmark'],
                model=item['model'],
                value=item['mean'] if 'mean' in item else item['value'],
                timestamp=item.get('timestamp', ''),
                hostname=item.get('hostname', ''),
            )
            self.store_result(result)
            count += 1

        return count

    def import_csv(self, input_path: str) -> int:
        """Import results from CSV file."""
        count = 0

        with open(input_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                result = BenchmarkResult(
                    benchmark=row['benchmark'],
                    model=row['model'],
                    value=float(row.get('mean', row.get('value', 0))),
                    timestamp=row.get('timestamp', ''),
                )
                self.store_result(result)
                count += 1

        return count

    # --- Utilities ---

    def get_benchmarks(self) -> List[str]:
        """Get list of unique benchmark names."""
        cursor = self.conn.execute(
            "SELECT DISTINCT benchmark FROM results ORDER BY benchmark"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_models(self) -> List[str]:
        """Get list of unique model names."""
        cursor = self.conn.execute(
            "SELECT DISTINCT model FROM results ORDER BY model"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_result_count(self) -> int:
        """Get total number of results."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM results")
        return cursor.fetchone()[0]

    def vacuum(self):
        """Optimize database file."""
        self.conn.execute("VACUUM")


def detect_gpu_info() -> Dict[str, str]:
    """Try to detect GPU information from the system."""
    info = {"gpu_name": "", "gpu_arch": ""}

    # Try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 1:
                info["gpu_name"] = parts[0]
            if len(parts) >= 2:
                info["gpu_arch"] = f"sm_{parts[1].replace('.', '')}"
    except:
        pass

    # Try rocm-smi for AMD
    if not info["gpu_name"]:
        try:
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Card series' in line:
                        info["gpu_name"] = line.split(':')[-1].strip()
                        break
        except:
            pass

    return info


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return ""


# CLI for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HeCBench Results Database CLI")
    parser.add_argument("--db", default="hecbench_results.db",
                        help="Database file path")

    subparsers = parser.add_subparsers(dest="command")

    # query command
    query_parser = subparsers.add_parser("query", help="Query results")
    query_parser.add_argument("-b", "--benchmark", help="Filter by benchmark")
    query_parser.add_argument("-m", "--model", help="Filter by model")
    query_parser.add_argument("--since", help="Filter by date (YYYY-MM-DD)")
    query_parser.add_argument("-n", "--limit", type=int, help="Limit results")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("-b", "--benchmark", help="Filter by benchmark")
    stats_parser.add_argument("-m", "--model", help="Filter by model")

    # export command
    export_parser = subparsers.add_parser("export", help="Export results")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("-f", "--format", choices=["json", "csv"],
                               default="csv", help="Output format")
    export_parser.add_argument("-b", "--benchmark", help="Filter by benchmark")
    export_parser.add_argument("-m", "--model", help="Filter by model")

    # import command
    import_parser = subparsers.add_parser("import", help="Import results")
    import_parser.add_argument("input", help="Input file path")

    # info command
    info_parser = subparsers.add_parser("info", help="Show database info")

    args = parser.parse_args()

    db = ResultsDB(args.db)

    if args.command == "query":
        results = db.query(
            benchmark=args.benchmark,
            model=args.model,
            since=args.since,
            limit=args.limit
        )
        for r in results:
            print(f"{r['benchmark']}-{r['model']}: {r['value']} {r['unit']} "
                  f"({r['timestamp']})")

    elif args.command == "stats":
        stats = db.get_stats(benchmark=args.benchmark, model=args.model)
        print(f"{'Benchmark':<30} {'Model':<8} {'Count':>6} {'Mean':>12} "
              f"{'Median':>12} {'StdDev':>10}")
        print("-" * 90)
        for s in stats:
            print(f"{s.benchmark:<30} {s.model:<8} {s.count:>6} "
                  f"{s.mean:>12.4f} {s.median:>12.4f} {s.std_dev:>10.4f}")

    elif args.command == "export":
        if args.format == "json":
            count = db.export_json(args.output, args.benchmark, args.model)
        else:
            count = db.export_csv(args.output, args.benchmark, args.model)
        print(f"Exported {count} results to {args.output}")

    elif args.command == "import":
        if args.input.endswith(".json"):
            count = db.import_json(args.input)
        else:
            count = db.import_csv(args.input)
        print(f"Imported {count} results from {args.input}")

    elif args.command == "info":
        print(f"Database: {db.db_path}")
        print(f"Total results: {db.get_result_count()}")
        print(f"Benchmarks: {len(db.get_benchmarks())}")
        print(f"Models: {', '.join(db.get_models())}")

    else:
        parser.print_help()

    db.close()
