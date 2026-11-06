import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ASYNC_EVENTS = []


def receive_response(reader, expected_id=None):
    while True:
        line = reader.readline()
        if not line:
            raise RuntimeError("Server closed connection before responding")

        response = json.loads(line)

        if expected_id is not None and response.get("request_id") != expected_id:
            if "event" in response:
                ASYNC_EVENTS.append(response)
                continue
            raise AssertionError(response)

        return response


def send_request(client, reader, command, payload, request_id):
    request = {
        "protocol_version": 1,
        "request_id": request_id,
        "command": command,
        "payload": payload,
    }

    client.sendall((json.dumps(request) + "\n").encode())
    response = receive_response(reader, request_id)

    assert response.get("ok") is True, response
    return response["result"]


def main():
    executable = str(Path(sys.argv[1]).resolve())

    with tempfile.TemporaryDirectory(prefix="spiking-ipc-test-") as temp:
        socket_path = str(Path(temp) / "server.sock")
        log_path = Path(temp) / "server.log"

        with log_path.open("w+") as log:
            server = subprocess.Popen(
                [
                    executable,
                    "--server",
                    "--small",
                    "--socket",
                    socket_path,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )

            client = None

            try:
                # Wait for the server socket to become available.
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"Server exited early with code {server.returncode}"
                        )

                    if os.path.exists(socket_path):
                        try:
                            client = socket.socket(
                                socket.AF_UNIX, socket.SOCK_STREAM
                            )
                            client.settimeout(5.0)
                            client.connect(socket_path)
                            break
                        except OSError:
                            client.close()
                            client = None

                    time.sleep(0.05)
                else:
                    raise TimeoutError("Server socket did not become ready")

                # Verify socket permissions.
                mode = stat.S_IMODE(os.stat(socket_path).st_mode)
                assert mode == 0o600, oct(mode)

                reader = client.makefile("r", encoding="utf-8")

                # A stream request may arrive in multiple writes.
                fragmented = {
                    "protocol_version": 1,
                    "request_id": "fragmented",
                    "command": "get_status",
                    "payload": {},
                }
                fragmented_bytes = (
                    json.dumps(fragmented) + "\n"
                ).encode()
                split_at = len(fragmented_bytes) // 2
                client.sendall(fragmented_bytes[:split_at])
                time.sleep(0.05)
                client.sendall(fragmented_bytes[split_at:])

                fragmented_response = receive_response(
                    reader, "fragmented"
                )
                assert fragmented_response["ok"] is True

                # Multiple requests may arrive in one write.
                pipelined = [
                    {
                        "protocol_version": 1,
                        "request_id": "pipeline-status",
                        "command": "get_status",
                        "payload": {},
                    },
                    {
                        "protocol_version": 1,
                        "request_id": "pipeline-telemetry",
                        "command": "get_telemetry",
                        "payload": {},
                    },
                ]
                pipeline_bytes = b"".join(
                    (json.dumps(request) + "\n").encode()
                    for request in pipelined
                )
                client.sendall(pipeline_bytes)

                pipeline_status = receive_response(
                    reader, "pipeline-status"
                )
                pipeline_telemetry = receive_response(
                    reader, "pipeline-telemetry"
                )
                assert pipeline_status["ok"] is True
                assert pipeline_telemetry["ok"] is True

                status = send_request(
                    client, reader, "get_status", {}, "status"
                )
                assert status["status"] == "ready", status

                configuration = send_request(
                    client, reader, "get_parameters", {}, "parameters"
                )
                assert "requested" in configuration
                assert "active" in configuration

                # Configure a deterministic two-neuron network so this
                # integration test exercises real spike events over IPC.
                update = send_request(
                    client, reader, "apply_parameters",
                    {
                        "parameters": {
                            "simulation": {
                                "excitatory_count": 1,
                                "inhibitory_count": 1,
                                "steps": 1,
                                "dt": 0.001,
                                "threshold": -64.9995,
                                "excitatory_current_min": 4.0,
                                "excitatory_current_max": 4.0,
                                "inhibitory_current_min": 4.0,
                                "inhibitory_current_max": 4.0,
                                "initial_voltage_min": -65.0,
                                "initial_voltage_max": -65.0,
                            },
                            "topology": {
                                "p_ee": 0.0,
                                "p_ei": 0.0,
                                "p_ie": 0.0,
                                "p_ii": 0.0,
                            },
                        }
                    },
                    "spike-test-parameters",
                )
                assert update["accepted"] is True, update
                assert update["requires_reinitialization"] is True, update

                reinitialized = send_request(
                    client, reader, "reinitialize", {},
                    "spike-test-reinitialize"
                )
                assert reinitialized["telemetry"]["completed_steps"] == "0"

                # The dashboard's connectome panel reads statistics from the
                # active CSR graph, including population-pair edge counts.
                connectivity = send_request(
                    client, reader, "get_connectivity", {}, "connectivity"
                )
                assert connectivity["neuron_count"] == 2, connectivity
                assert connectivity["excitatory_count"] == 1, connectivity
                assert connectivity["inhibitory_count"] == 1, connectivity
                assert connectivity["synapse_count"] == 0, connectivity
                assert connectivity["density"] == 0.0, connectivity
                assert connectivity["population_edges"] == {
                    "excitatory_to_excitatory": 0,
                    "excitatory_to_inhibitory": 0,
                    "inhibitory_to_excitatory": 0,
                    "inhibitory_to_inhibitory": 0,
                }, connectivity
                assert connectivity["outgoing_degree"]["mean"] == 0.0
                assert connectivity["incoming_degree"]["mean"] == 0.0

                telemetry = send_request(
                    client, reader, "get_telemetry", {}, "telemetry"
                )
                assert telemetry["completed_steps"] == "0"

                spike_snapshot = send_request(
                    client, reader, "get_spike_events", {},
                    "spike-events-initial"
                )
                assert spike_snapshot["first_sequence"] == "0"
                assert spike_snapshot["total_events"] == "0"
                assert spike_snapshot["dropped_events"] == "0"
                assert spike_snapshot["events"] == []

                spike_subscription = send_request(
                    client, reader, "subscribe_spike_events", {},
                    "subscribe-spikes"
                )
                assert spike_subscription["subscribed"] is True

                subscription = send_request(
                    client, reader, "subscribe_telemetry", {},
                    "subscribe"
                )
                assert subscription["subscribed"] is True
                assert subscription["telemetry"]["completed_steps"] == "0"

                stepped = send_request(
                    client, reader, "step", {"steps": 1}, "step"
                )
                assert stepped["telemetry"]["completed_steps"] == "1"

                # Telemetry is intentionally prioritized over spike-event
                # notifications. Drain both asynchronous message types rather
                # than assuming the spike notification arrives first.
                telemetry_received = False
                spike_payload_with_events = None
                while not telemetry_received or spike_payload_with_events is None:
                    event_line = reader.readline()
                    assert event_line, "Server closed before asynchronous updates"
                    event = json.loads(event_line)
                    if event.get("event") == "spike_events":
                        ASYNC_EVENTS.append(event)
                        assert "events" in event["spike_events"], event
                        if len(event["spike_events"]["events"]) == 2:
                            spike_payload_with_events = event["spike_events"]
                        continue
                    assert event["protocol_version"] == 1, event
                    assert event["event"] == "telemetry_update", event
                    assert event["telemetry"]["completed_steps"] == "1", event
                    assert (
                        event["telemetry"]["sequence"]
                        == stepped["telemetry"]["sequence"]
                    ), event
                    telemetry_received = True

                notification_payload = spike_payload_with_events
                assert len(notification_payload["events"]) == 2, (
                    notification_payload
                )
                assert {
                    item["neuron_id"]
                    for item in notification_payload["events"]
                } == {0, 1}, notification_payload
                assert {
                    item["step"] for item in notification_payload["events"]
                } == {"1"}, notification_payload
                assert {
                    item["sequence"] for item in notification_payload["events"]
                } == {"0", "1"}, notification_payload

                # Polling exposes the same sequenced event snapshot used by
                # asynchronous spike_events notifications.
                spike_snapshot_after_step = send_request(
                    client, reader, "get_spike_events", {},
                    "spike-events-after-step"
                )
                assert "events" in spike_snapshot_after_step
                assert "simulation_generation" in spike_snapshot_after_step
                assert len(spike_snapshot_after_step["events"]) == 2
                assert {
                    item["neuron_id"]
                    for item in spike_snapshot_after_step["events"]
                } == {0, 1}
                assert {
                    item["step"] for item in spike_snapshot_after_step["events"]
                } == {"1"}
                for spike_event_message in ASYNC_EVENTS:
                    if spike_event_message.get("event") == "spike_events":
                        payload = spike_event_message["spike_events"]
                        assert "first_sequence" in payload
                        assert "total_events" in payload
                        assert "dropped_events" in payload
                        assert "simulation_generation" in payload
                        assert isinstance(payload["events"], list)

                spike_unsubscription = send_request(
                    client, reader, "unsubscribe_spike_events", {},
                    "unsubscribe-spikes"
                )
                assert spike_unsubscription["subscribed"] is False

                unsubscription = send_request(
                    client, reader, "unsubscribe_telemetry", {},
                    "unsubscribe"
                )
                assert unsubscription["subscribed"] is False

                # Invalid JSON must return a transport error without
                # terminating the connection.
                client.sendall(b"{not valid json}\n")
                malformed = receive_response(reader)
                assert malformed["ok"] is False
                assert malformed["error"]["code"] == "invalid_json"

                # Unknown commands must return a structured error.
                client.sendall((
                    json.dumps({
                        "protocol_version": 1,
                        "request_id": "unknown",
                        "command": "not_a_real_command",
                        "payload": {},
                    }) + "\n"
                ).encode())

                unknown = receive_response(reader, "unknown")
                assert unknown["ok"] is False
                assert unknown["error"]["code"] == "unknown_command"

                # Oversized lines must be rejected.
                client.sendall(b"x" * (1024 * 1024 + 1))
                oversized = receive_response(reader)
                assert oversized["ok"] is False
                assert oversized["error"]["code"] == "request_too_large"

                # The oversized request closes its connection. Reconnect
                # and leave this new connection genuinely idle.
                reader.close()
                client.close()
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(5.0)
                client.connect(socket_path)
                reader = client.makefile("r", encoding="utf-8")
                time.sleep(0.15)

                # Shutdown must not wait indefinitely for client data.
                server.send_signal(signal.SIGTERM)
                server.wait(timeout=5.0)
                assert server.returncode == 0, server.returncode

                assert not os.path.exists(socket_path), (
                    "Server did not remove its socket on shutdown"
                )

                print("IPC integration test passed.")
                print("  Socket permissions: 0600")
                print("  Status/configuration/telemetry: passed")
                print("  Telemetry and spike-event subscription acknowledgements: passed")
                print("  Spike-event snapshot protocol: passed")
                print("  Asynchronous telemetry event: passed")
                print("  Simulation step: passed")
                print("  Invalid JSON handling: passed")
                print("  Unknown command handling: passed")
                print("  Oversized request handling: passed")
                print("  Shutdown with idle client: passed")
                print("  Socket cleanup: passed")

            finally:
                if client is not None:
                    client.close()

                if server.poll() is None:
                    server.send_signal(signal.SIGTERM)
                    try:
                        server.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()


if __name__ == "__main__":
    main()
