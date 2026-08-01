import Foundation

public struct HermesGatewayEvent: Equatable, Sendable {
    public let type: String
    public let sessionID: String?
    public let payload: [String: JSONValue]

    public init(type: String, sessionID: String?, payload: [String: JSONValue]) {
        self.type = type
        self.sessionID = sessionID
        self.payload = payload
    }
}

public enum HermesGatewayClientError: Error, LocalizedError, Equatable {
    case disconnected
    case malformedResponse
    case readinessTimedOut
    case rpcError
    case sendFailed

    public var errorDescription: String? {
        switch self {
        case .disconnected: "Hermes gateway disconnected."
        case .malformedResponse: "Hermes returned a malformed response."
        case .readinessTimedOut: "Hermes gateway did not become ready in time."
        case .rpcError: "Hermes RPC request failed."
        case .sendFailed: "Hermes request could not be sent."
        }
    }
}

@MainActor
public protocol HermesGatewayClientProtocol: AnyObject {
    var onEvent: ((HermesGatewayEvent) -> Void)? { get set }
    var onDisconnect: ((String) -> Void)? { get set }
    func connectAndWaitUntilReady(timeoutSeconds: Double) async throws
    func call(method: String, params: [String: JSONValue]) async throws -> JSONValue
    func close()
}

public typealias HermesGatewayClientFactory = @MainActor (URL) -> any HermesGatewayClientProtocol

/// JSON-RPC transport for the ephemeral, authenticated loopback sidecar.
@MainActor
public final class URLSessionHermesGatewayClient: HermesGatewayClientProtocol {
    private let task: URLSessionWebSocketTask
    private var nextID = 1
    private var pending: [Int: CheckedContinuation<JSONValue, Error>] = [:]
    private var readinessWaiters: [UUID: CheckedContinuation<Void, Error>] = [:]
    private var readinessTimeouts: [UUID: Task<Void, Never>] = [:]
    private var started = false
    private var ready = false
    private var terminated = false

    public var onEvent: ((HermesGatewayEvent) -> Void)?
    public var onDisconnect: ((String) -> Void)?

    public init(url: URL, session: URLSession = .shared) {
        task = session.webSocketTask(with: url)
    }

    /// Compatibility for the pre-readiness store lifecycle. New callers wait.
    public func connect() {
        startIfNeeded()
    }

    public func connectAndWaitUntilReady(timeoutSeconds: Double) async throws {
        guard !terminated else { throw HermesGatewayClientError.disconnected }
        if ready { return }
        let timeout = min(max(timeoutSeconds, 0.01), 10)
        let waiterID = UUID()
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            guard !self.terminated else {
                continuation.resume(throwing: HermesGatewayClientError.disconnected)
                return
            }
            if self.ready {
                continuation.resume()
                return
            }
            // Install before task.resume(): a fast ready event must not race us.
            self.readinessWaiters[waiterID] = continuation
            self.readinessTimeouts[waiterID] = Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                self?.timeoutReadinessWaiter(waiterID)
            }
            self.startIfNeeded()
        }
    }

    public func call(method: String, params: [String: JSONValue] = [:]) async throws -> JSONValue {
        guard started, !terminated else { throw HermesGatewayClientError.disconnected }
        let id = nextID
        nextID += 1
        let request: [String: JSONValue] = [
            "jsonrpc": .string("2.0"),
            "id": .number(Double(id)),
            "method": .string(method),
            "params": .object(params),
        ]
        let data = try JSONEncoder().encode(request)
        guard let text = String(data: data, encoding: .utf8) else {
            throw HermesGatewayClientError.malformedResponse
        }
        return try await withCheckedThrowingContinuation { continuation in
            guard !self.terminated else {
                continuation.resume(throwing: HermesGatewayClientError.disconnected)
                return
            }
            self.pending[id] = continuation
            self.task.send(.string(text)) { [weak self] error in
                guard error != nil else { return }
                Task { @MainActor [weak self] in
                    self?.failPendingRequest(id, error: .sendFailed)
                }
            }
        }
    }

    public func close() {
        terminate(error: .disconnected, notify: false)
        task.cancel(with: .goingAway, reason: nil)
    }

    private func startIfNeeded() {
        guard !started, !terminated else { return }
        started = true
        task.resume()
        receiveNext()
    }

    private func receiveNext() {
        guard !terminated else { return }
        task.receive { [weak self] result in
            Task { @MainActor [weak self] in
                guard let self, !self.terminated else { return }
                switch result {
                case .success(let message):
                    self.handle(message)
                    self.receiveNext()
                case .failure:
                    self.terminate(error: .disconnected, notify: true)
                }
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let data: Data
        switch message {
        case .string(let text): data = Data(text.utf8)
        case .data(let received): data = received
        @unknown default:
            terminate(error: .malformedResponse, notify: true)
            return
        }
        guard let root = try? JSONDecoder().decode([String: JSONValue].self, from: data) else {
            terminate(error: .malformedResponse, notify: true)
            return
        }

        if root["id"] != nil {
            guard let id = rpcID(root["id"]), root["jsonrpc"]?.stringValue == "2.0" else {
                terminate(error: .malformedResponse, notify: true)
                return
            }
            if root["error"] != nil {
                guard object(from: root["error"]) != nil else {
                    terminate(error: .malformedResponse, notify: true)
                    return
                }
                failPendingRequest(id, error: .rpcError)
                return
            }
            guard let result = root["result"] else {
                terminate(error: .malformedResponse, notify: true)
                return
            }
            pending.removeValue(forKey: id)?.resume(returning: result)
            return
        }

        guard root["method"]?.stringValue == "event",
              let params = object(from: root["params"]),
              let type = params["type"]?.stringValue
        else {
            terminate(error: .malformedResponse, notify: true)
            return
        }
        let event = HermesGatewayEvent(
            type: type,
            sessionID: params["session_id"]?.stringValue,
            payload: object(from: params["payload"]) ?? [:]
        )
        onEvent?(event)
        if type == "gateway.ready" { finishReadiness() }
    }

    private func rpcID(_ value: JSONValue?) -> Int? {
        guard case .number(let number) = value,
              number.isFinite,
              number.rounded() == number,
              number >= 0,
              number <= Double(Int.max)
        else { return nil }
        return Int(number)
    }

    private func object(from value: JSONValue?) -> [String: JSONValue]? {
        guard case .object(let object) = value else { return nil }
        return object
    }

    private func finishReadiness() {
        guard !ready else { return }
        ready = true
        let waiters = readinessWaiters
        readinessWaiters.removeAll()
        let timeouts = readinessTimeouts
        readinessTimeouts.removeAll()
        timeouts.values.forEach { $0.cancel() }
        waiters.values.forEach { $0.resume() }
    }

    private func timeoutReadinessWaiter(_ id: UUID) {
        readinessTimeouts.removeValue(forKey: id)
        readinessWaiters.removeValue(forKey: id)?.resume(throwing: HermesGatewayClientError.readinessTimedOut)
    }

    private func failPendingRequest(_ id: Int, error: HermesGatewayClientError) {
        pending.removeValue(forKey: id)?.resume(throwing: error)
    }

    private func terminate(error: HermesGatewayClientError, notify: Bool) {
        guard !terminated else { return }
        terminated = true
        let rpcWaiters = pending
        pending.removeAll()
        let readyWaiters = readinessWaiters
        readinessWaiters.removeAll()
        let timeouts = readinessTimeouts
        readinessTimeouts.removeAll()
        timeouts.values.forEach { $0.cancel() }
        rpcWaiters.values.forEach { $0.resume(throwing: error) }
        readyWaiters.values.forEach { $0.resume(throwing: error) }
        if notify {
            onDisconnect?(error.errorDescription ?? "Hermes gateway disconnected.")
        }
    }
}
