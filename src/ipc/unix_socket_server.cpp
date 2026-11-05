#include "ipc/unix_socket_server.h"

#include "ipc/protocol.h"

#include <nlohmann/json.hpp>

#include <algorithm>

#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <deque>
#include <optional>
#include <iostream>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

namespace spiking::ipc {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaximumLineBytes = 1024 * 1024;

volatile std::sig_atomic_t g_stop_requested = 0;

void stop_signal_handler(int) {
    g_stop_requested = 1;
}

class SignalGuard {
public:
    SignalGuard() {
        struct sigaction action {};
        action.sa_handler = stop_signal_handler;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;

        if (sigaction(SIGINT, &action, &old_int_) != 0) {
            throw std::runtime_error(
                std::string("sigaction(SIGINT) failed: ") + std::strerror(errno));
        }

        if (sigaction(SIGTERM, &action, &old_term_) != 0) {
            const int saved_errno = errno;
            sigaction(SIGINT, &old_int_, nullptr);
            throw std::runtime_error(
                std::string("sigaction(SIGTERM) failed: ") +
                std::strerror(saved_errno));
        }

        g_stop_requested = 0;
        installed_ = true;
    }

    ~SignalGuard() {
        if (installed_) {
            sigaction(SIGINT, &old_int_, nullptr);
            sigaction(SIGTERM, &old_term_, nullptr);
        }
    }

    SignalGuard(const SignalGuard&) = delete;
    SignalGuard& operator=(const SignalGuard&) = delete;

private:
    struct sigaction old_int_ {};
    struct sigaction old_term_ {};
    bool installed_ = false;
};

class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) : fd_(fd) {}
    ~FileDescriptor() {
        if (fd_ >= 0)
            close(fd_);
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    FileDescriptor(FileDescriptor&& other) noexcept
        : fd_(std::exchange(other.fd_, -1)) {}

    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            if (fd_ >= 0)
                close(fd_);
            fd_ = std::exchange(other.fd_, -1);
        }
        return *this;
    }

    int get() const noexcept { return fd_; }

private:
    int fd_;
};

sockaddr_un make_address(const std::string& path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;

    if (path.empty() || path.size() >= sizeof(address.sun_path)) {
        throw std::invalid_argument(
            "Unix socket path is empty or too long for sockaddr_un");
    }

    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    return address;
}

void remove_stale_socket_if_safe(const std::string& path,
                                 const sockaddr_un& address) {
    struct stat info {};
    if (lstat(path.c_str(), &info) != 0) {
        if (errno == ENOENT)
            return;
        throw std::runtime_error(
            std::string("lstat socket path failed: ") + std::strerror(errno));
    }

    if (!S_ISSOCK(info.st_mode)) {
        throw std::runtime_error(
            "socket path already exists and is not a socket; refusing to remove it");
    }

    FileDescriptor probe(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (probe.get() < 0) {
        throw std::runtime_error(
            std::string("socket probe creation failed: ") + std::strerror(errno));
    }

    if (connect(probe.get(),
                reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) == 0) {
        throw std::runtime_error("another server is already listening on this socket");
    }

    const int connect_error = errno;
    if (connect_error != ECONNREFUSED && connect_error != ENOENT) {
        throw std::runtime_error(
            std::string("cannot safely determine whether socket is stale: ") +
            std::strerror(connect_error));
    }

    if (unlink(path.c_str()) != 0 && errno != ENOENT) {
        throw std::runtime_error(
            std::string("could not remove stale socket: ") + std::strerror(errno));
    }
}

constexpr std::size_t kMaximumQueuedResponseBytes = 4 * 1024 * 1024;

bool enqueue_json_line(
    std::deque<std::string>& responses,
    std::size_t& queued_bytes,
    const Json& message
) {
    std::string line = message.dump();
    line.push_back('\n');

    if (line.size() > kMaximumQueuedResponseBytes ||
        queued_bytes > kMaximumQueuedResponseBytes - line.size()) {
        return false;
    }

    queued_bytes += line.size();
    responses.push_back(std::move(line));
    return true;
}

Json transport_error(const std::string& code, const std::string& message) {
    return {
        {"protocol_version", kProtocolVersion},
        {"request_id", nullptr},
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };
}

bool process_line(
    const std::string& line,
    SimulationController& controller,
    std::deque<std::string>& responses,
    std::size_t& queued_bytes,
    bool& subscribed,
    std::uint64_t& last_telemetry_sequence,
    bool& spike_events_subscribed,
    bool& spike_subscription_initialized
) {
    try {
        const Json request = Json::parse(line);
        const Json response = handle_request(controller, request);

        // Subscription state is connection-local. Only change it after
        // the protocol layer has accepted the command.
        if (response.value("ok", false) &&
            request.is_object() &&
            request.contains("command") &&
            request.at("command").is_string()) {
            const std::string command =
                request.at("command").get<std::string>();

            if (command == "subscribe_telemetry") {
                subscribed = true;

                // Begin with updates published after this subscription.
                last_telemetry_sequence =
                    controller.telemetry_snapshot().sequence;
            } else if (command == "unsubscribe_telemetry") {
                subscribed = false;
            } else if (command == "subscribe_spike_events") {
                spike_events_subscribed = true;
                spike_subscription_initialized = false;
            } else if (command == "unsubscribe_spike_events") {
                spike_events_subscribed = false;
                spike_subscription_initialized = false;
            }
        }

        return enqueue_json_line(responses, queued_bytes, response);
    } catch (const Json::parse_error&) {
        return enqueue_json_line(
            responses, queued_bytes,
            transport_error("invalid_json", "request line is not valid JSON"));
    } catch (const std::exception& error) {
        return enqueue_json_line(
            responses, queued_bytes,
            transport_error("transport_error", error.what()));
    }
}

void serve_client(int client_fd, SimulationController& controller) {
    std::string pending;
    char buffer[8192];

    std::deque<std::string> responses;
    std::size_t queued_response_bytes = 0;

    // A telemetry notification is deliberately not placed in the response
    // queue. Only the newest unsent snapshot is retained.
    std::optional<std::string> pending_telemetry;
    std::optional<std::string> pending_spike_events;
    std::string current_output;
    std::size_t current_output_offset = 0;

    bool subscribed = false;
    bool spike_events_subscribed = false;
    bool spike_subscription_initialized = false;
    bool close_after_flush = false;
    std::uint64_t last_telemetry_sequence = 0;
    std::uint64_t last_spike_event_sequence = 0;
    std::uint64_t last_spike_event_generation = 0;
    std::uint64_t last_spike_dropped_events = 0;

    while (!g_stop_requested) {
        if (subscribed) {
            const auto snapshot = controller.telemetry_snapshot();

            if (snapshot.sequence > last_telemetry_sequence) {
                pending_telemetry =
                    telemetry_event_json(snapshot).dump() + "\n";
                last_telemetry_sequence = snapshot.sequence;
            }
        }

        if (spike_events_subscribed && !pending_spike_events.has_value()) {
            constexpr std::size_t kMaximumSpikeEventsPerUpdate = 8192;
            const auto current =
                controller.spike_events_snapshot(last_spike_event_sequence, 0);
            const bool generation_changed =
                current.simulation_generation != last_spike_event_generation;
            const bool initialize_stream =
                !spike_subscription_initialized || generation_changed;

            if (initialize_stream) {
                last_spike_event_generation = current.simulation_generation;
                last_spike_event_sequence = 0;
            }

            // The dashboard only renders a short rolling time window. Never
            // replay an old backlog: begin with the newest bounded slice on
            // subscription/reinitialization, and jump forward if production
            // outruns the IPC consumer.
            std::uint64_t first_sequence = std::max(
                last_spike_event_sequence, current.first_sequence);
            const std::uint64_t newest_slice_start =
                current.total_events > kMaximumSpikeEventsPerUpdate
                    ? current.total_events - kMaximumSpikeEventsPerUpdate
                    : 0;
            const bool consumer_is_behind =
                current.total_events > first_sequence &&
                current.total_events - first_sequence >
                    kMaximumSpikeEventsPerUpdate;

            if (initialize_stream || consumer_is_behind) {
                first_sequence = std::max(first_sequence, newest_slice_start);
            }

            const auto event_snapshot = controller.spike_events_snapshot(
                first_sequence, kMaximumSpikeEventsPerUpdate);

            if (initialize_stream || !event_snapshot.events.empty() ||
                event_snapshot.dropped_events != last_spike_dropped_events) {
                pending_spike_events =
                    spike_events_event_json(event_snapshot).dump() + "\n";
                if (!event_snapshot.events.empty()) {
                    last_spike_event_sequence =
                        event_snapshot.events.back().sequence + 1;
                } else {
                    last_spike_event_sequence = event_snapshot.total_events;
                }
                last_spike_dropped_events = event_snapshot.dropped_events;
            }

            if (initialize_stream) {
                spike_subscription_initialized = true;
            }
        }

        // Finish any partially written frame before starting another.
        // Command responses take priority over an unsent telemetry update.
        if (current_output.empty()) {
            if (!responses.empty()) {
                current_output = std::move(responses.front());
                responses.pop_front();

                queued_response_bytes -= current_output.size();
            } else if (pending_telemetry.has_value()) {
                // Telemetry is latency-sensitive and must not be starved by a
                // continuous stream of spike-event notifications.
                current_output = std::move(*pending_telemetry);
                pending_telemetry.reset();
            } else if (pending_spike_events.has_value()) {
                current_output = std::move(*pending_spike_events);
                pending_spike_events.reset();
            }

            current_output_offset = 0;
        }

        if (close_after_flush &&
            current_output.empty() &&
            responses.empty()) {
            return;
        }

        struct pollfd descriptor {};
        descriptor.fd = client_fd;
        descriptor.events = close_after_flush ? 0 : POLLIN;

        if (!current_output.empty())
            descriptor.events |= POLLOUT;

        const int poll_result = poll(&descriptor, 1, 200);

        if (poll_result == 0)
            continue;

        if (poll_result < 0) {
            if (errno == EINTR) {
                if (g_stop_requested)
                    return;
                continue;
            }
            return;
        }

        if (descriptor.revents & (POLLERR | POLLNVAL))
            return;

        // Once we're flushing a final error response, a disconnected
        // client cannot receive it. Avoid spinning on a persistent HUP.
        if (close_after_flush && (descriptor.revents & POLLHUP))
            return;

        if (!close_after_flush &&
            (descriptor.revents & (POLLIN | POLLHUP))) {
            const ssize_t received =
                recv(client_fd, buffer, sizeof(buffer), 0);

            if (received == 0)
                return;

            if (received < 0) {
                if (errno == EINTR || errno == EAGAIN ||
                    errno == EWOULDBLOCK) {
                    continue;
                }
                return;
            }

            pending.append(buffer, static_cast<std::size_t>(received));

            for (;;) {
                const std::size_t newline = pending.find('\n');

                if (newline == std::string::npos) {
                    if (pending.size() > kMaximumLineBytes) {
                        if (!enqueue_json_line(
                                responses, queued_response_bytes,
                                transport_error(
                                    "request_too_large",
                                    "request line exceeds the 1 MiB limit"))) {
                            return;
                        }

                        close_after_flush = true;
                    }
                    break;
                }

                if (newline > kMaximumLineBytes) {
                    if (!enqueue_json_line(
                            responses, queued_response_bytes,
                            transport_error(
                                "request_too_large",
                                "request line exceeds the 1 MiB limit"))) {
                        return;
                    }

                    close_after_flush = true;
                    break;
                }

                std::string line = pending.substr(0, newline);
                pending.erase(0, newline + 1);

                if (!line.empty() && line.back() == '\r')
                    line.pop_back();

                if (!process_line(
                        line, controller, responses,
                        queued_response_bytes, subscribed,
                        last_telemetry_sequence, spike_events_subscribed,
                        spike_subscription_initialized)) {
                    // A client that stops reading must not be able to
                    // accumulate an unbounded response backlog.
                    return;
                }
            }
        }

        if ((descriptor.revents & POLLOUT) &&
            !current_output.empty()) {
            const ssize_t sent = send(
                client_fd,
                current_output.data() + current_output_offset,
                current_output.size() - current_output_offset,
                MSG_NOSIGNAL | MSG_DONTWAIT);

            if (sent > 0) {
                current_output_offset +=
                    static_cast<std::size_t>(sent);

                if (current_output_offset == current_output.size()) {
                    current_output.clear();
                    current_output_offset = 0;
                    }
            } else if (sent == 0) {
                return;
            } else if (sent < 0 &&
                       errno != EINTR &&
                       errno != EAGAIN &&
                       errno != EWOULDBLOCK) {
                return;
            }
        }
    }
}

} // namespace

int run_unix_socket_server(
    SimulationController& controller,
    const std::string& socket_path
) {
    SignalGuard signals;
    const sockaddr_un address = make_address(socket_path);

    remove_stale_socket_if_safe(socket_path, address);

    FileDescriptor listener(socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0));
    if (listener.get() < 0) {
        throw std::runtime_error(
            std::string("socket creation failed: ") + std::strerror(errno));
    }

    const mode_t old_mask = umask(0077);
    const int bind_result = bind(
        listener.get(),
        reinterpret_cast<const sockaddr*>(&address),
        sizeof(address));
    const int bind_error = errno;
    umask(old_mask);

    if (bind_result != 0) {
        throw std::runtime_error(
            std::string("bind failed: ") + std::strerror(bind_error));
    }

    struct stat socket_info {};
    if (lstat(socket_path.c_str(), &socket_info) != 0) {
        const int saved_errno = errno;
        unlink(socket_path.c_str());
        throw std::runtime_error(
            std::string("lstat after bind failed: ") + std::strerror(saved_errno));
    }

    const dev_t socket_device = socket_info.st_dev;
    const ino_t socket_inode = socket_info.st_ino;

    if (chmod(socket_path.c_str(), S_IRUSR | S_IWUSR) != 0) {
        const int saved_errno = errno;
        unlink(socket_path.c_str());
        throw std::runtime_error(
            std::string("chmod socket failed: ") + std::strerror(saved_errno));
    }

    if (listen(listener.get(), 1) != 0) {
        const int saved_errno = errno;
        unlink(socket_path.c_str());
        throw std::runtime_error(
            std::string("listen failed: ") + std::strerror(saved_errno));
    }

    std::cout << "IPC server listening on " << socket_path << '\n'
              << "Socket permissions: 0600\n"
              << "Protocol version: " << kProtocolVersion << '\n'
              << "Press Ctrl+C to stop.\n"
              << std::flush;

    while (!g_stop_requested) {
        const int client = accept4(
            listener.get(), nullptr, nullptr, SOCK_CLOEXEC);

        if (client < 0) {
            if (errno == EINTR) {
                if (g_stop_requested)
                    break;
                continue;
            }

            if (g_stop_requested)
                break;

            std::cerr << "IPC accept failed: " << std::strerror(errno) << '\n';
            continue;
        }

        FileDescriptor client_fd(client);
        serve_client(client_fd.get(), controller);
    }

    struct stat current_info {};
    if (lstat(socket_path.c_str(), &current_info) == 0 &&
        S_ISSOCK(current_info.st_mode) &&
        current_info.st_dev == socket_device &&
        current_info.st_ino == socket_inode) {
        unlink(socket_path.c_str());
    }

    std::cout << "IPC server stopped.\n";
    return 0;
}

} // namespace spiking::ipc
