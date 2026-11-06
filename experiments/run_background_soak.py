#!/usr/bin/env python3
"""Run SpikingModel as a low-priority background IPC service and sample it."""

import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid


PROTOCOL_VERSION = 1


def request(stream, writer, command, payload=None):
    request_id = uuid.uuid4().hex
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "command": command,
        "payload": payload or {},
    }
    writer.write(json.dumps(message, separators=(",", ":")) + "\n")
    writer.flush()
    line = stream.readline()
    if not line:
        raise RuntimeError("IPC server closed the connection")
    response = json.loads(line)
    if response.get("request_id") != request_id:
        raise RuntimeError(f"IPC response ID mismatch: {response}")
    if not response.get("ok"):
        raise RuntimeError(f"IPC {command} failed: {response.get('error')}")
    return response.get("result", {})


def process_usage(pid):
    try:
        result = subprocess.run(
            ["ps", "-o", "%cpu=,rss=", "-p", str(pid)],
            check=False, capture_output=True, text=True, timeout=2,
        )
        fields = result.stdout.split()
        if len(fields) >= 2:
            return {"cpu_percent": float(fields[0]),
                    "rss_mib": int(fields[1]) / 1024.0}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"cpu_percent": None, "rss_mib": None}


def gpu_usage(device):
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits", "-i", str(device)],
            check=False, capture_output=True, text=True, timeout=3,
        )
        fields = [part.strip() for part in result.stdout.strip().split(",")]
        if result.returncode == 0 and len(fields) == 3:
            def number(value):
                try:
                    return float(value)
                except ValueError:
                    return None
            return {"gpu_utilization_percent": number(fields[0]),
                    "gpu_memory_mib": number(fields[1]),
                    "gpu_power_watts": number(fields[2])}
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"gpu_utilization_percent": None,
            "gpu_memory_mib": None, "gpu_power_watts": None}


def main():
    parser = argparse.ArgumentParser(
        description="Two-times-scale, two-minute-wall-time background soak test")
    parser.add_argument("--binary", default="./build-cuda/spiking_model")
    parser.add_argument("--config", default="experiments/background_soak_2x.json")
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nice", type=int, default=10,
                        help="CPU scheduling niceness for the server (default: 10)")
    parser.add_argument("--output",
                        default="experiments/logs/background_soak_2x_120s.jsonl")
    args = parser.parse_args()

    if args.duration_seconds <= 0 or args.sample_seconds <= 0:
        parser.error("duration and sample interval must be positive")

    binary = pathlib.Path(args.binary).resolve()
    config = pathlib.Path(args.config).resolve()
    output = pathlib.Path(args.output).resolve()
    if not binary.is_file():
        parser.error(f"simulator executable not found: {binary}")
    if not config.is_file():
        parser.error(f"configuration not found: {config}")

    output.parent.mkdir(parents=True, exist_ok=True)
    socket_path = os.path.join(
        tempfile.gettempdir(), f"spiking-model-soak-{os.getpid()}.sock")
    server_log_path = output.with_suffix(".server.log")
    run_id = dt.datetime.now(dt.timezone.utc).isoformat()
    records = []

    def record(kind, **values):
        item = {"run_id": run_id, "record_type": kind,
                "utc_time": dt.datetime.now(dt.timezone.utc).isoformat()}
        item.update(values)
        records.append(item)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")

    server_log = server_log_path.open("w", encoding="utf-8")
    process = None
    client = None
    try:
        def lower_cpu_priority():
            os.nice(args.nice)

        process = subprocess.Popen(
            [str(binary), "--server", "--enable-stdp", "--config", str(config),
             "--device", str(args.device), "--socket", socket_path],
            stdout=server_log, stderr=subprocess.STDOUT,
            preexec_fn=lower_cpu_priority,
        )

        deadline = time.monotonic() + 30.0
        while not os.path.exists(socket_path):
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited during startup with code {process.returncode}; "
                    f"inspect {server_log_path}")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"timed out waiting for IPC socket; inspect {server_log_path}")
            time.sleep(0.2)

        # This endpoint is a Unix-domain socket, not a TCP host/port pair.
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(10.0)
        client.connect(socket_path)
        reader = client.makefile("r", encoding="utf-8")
        writer = client.makefile("w", encoding="utf-8")

        status = request(reader, writer, "get_status")
        connectivity = request(reader, writer, "get_connectivity")
        started = request(reader, writer, "start")
        start_time = time.monotonic()
        record("started", pid=process.pid, nice=args.nice,
               duration_seconds=args.duration_seconds,
               sample_seconds=args.sample_seconds,
               status=status, start=started, connectivity=connectivity,
               process=process_usage(process.pid),
               gpu=gpu_usage(args.device))

        while True:
            elapsed = time.monotonic() - start_time
            remaining = args.duration_seconds - elapsed
            if remaining <= 0:
                break
            time.sleep(min(args.sample_seconds, remaining))
            if process.poll() is not None:
                raise RuntimeError(
                    f"background server exited with code {process.returncode}; "
                    f"inspect {server_log_path}")
            telemetry = request(reader, writer, "get_telemetry")
            record("sample", elapsed_wall_seconds=round(
                       time.monotonic() - start_time, 3),
                   telemetry=telemetry,
                   process=process_usage(process.pid),
                   gpu=gpu_usage(args.device))

        paused = request(reader, writer, "pause")
        final_telemetry = paused.get("telemetry") or request(
            reader, writer, "get_telemetry")
        final_connectivity = request(reader, writer, "get_connectivity")
        record("completed", elapsed_wall_seconds=round(
                   time.monotonic() - start_time, 3),
               pause=paused, final_telemetry=final_telemetry,
               final_connectivity=final_connectivity,
               process=process_usage(process.pid),
               gpu=gpu_usage(args.device))
        print(json.dumps({
            "result": "PASS",
            "wall_seconds": round(time.monotonic() - start_time, 3),
            "pid": process.pid,
            "connectivity_before": connectivity,
            "connectivity_after": final_connectivity,
            "final_telemetry": final_telemetry,
            "jsonl_log": str(output),
            "server_log": str(server_log_path),
        }, indent=2))
    finally:
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        server_log.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Soak interrupted; server shutdown requested.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"BACKGROUND SOAK FAILED: {error}", file=sys.stderr)
        sys.exit(1)
