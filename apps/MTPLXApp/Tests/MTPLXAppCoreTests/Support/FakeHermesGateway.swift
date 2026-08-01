import CryptoKit
import Foundation
import Network
@testable import MTPLXAppCore

/// A deliberately small loopback-only RFC 6455 fixture.  It supports exactly
/// the text frames the gateway client uses; it is not a general WebSocket
/// implementation and never listens beyond 127.0.0.1.
final class FakeHermesGateway: @unchecked Sendable {
    enum EventOnConnect {
        case gatewayReady
    }

    private let queue = DispatchQueue(label: "FakeHermesGateway")
    private let listener: NWListener
    private var port: NWEndpoint.Port?
    private var connection: NWConnection?
    private var handshakeBuffer = Data()
    private var frameBuffer = Data()
    private var handshakeComplete = false
    private var responses: [String: JSONValue] = [:]
    private var rpcErrors: Set<String> = []
    private var receivedMethods: [String] = []
    private var authenticated = false
    private let eventsOnConnect: [EventOnConnect]

    init(eventsOnConnect: [EventOnConnect]) throws {
        self.eventsOnConnect = eventsOnConnect
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        listener = try NWListener(using: parameters)
        let startup = FixtureStartupSignal()
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                startup.finish(failed: false)
            case .failed:
                startup.finish(failed: true)
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let gateway = self else { return }
            gateway.queue.async { gateway.accept(connection) }
        }
        listener.start(queue: queue)
        guard startup.wait(timeout: .now() + 1),
              !startup.failed,
              let port = listener.port
        else {
            throw FixtureError.startFailed
        }
        self.port = port
    }

    deinit {
        listener.cancel()
        connection?.cancel()
    }

    func webSocketURL(token: String) -> URL {
        return URL(string: "ws://127.0.0.1:\(port!.rawValue)/api/ws?token=\(token)")!
    }

    func respond(to method: String, result: JSONValue) {
        queue.sync { responses[method] = result }
    }

    func respondWithRPCError(to method: String) {
        _ = queue.sync { rpcErrors.insert(method) }
    }

    func sendEvent(type: String, sessionID: String?, payload: [String: JSONValue]) {
        queue.async {
            var params: [String: JSONValue] = [
                "type": .string(type),
                "payload": .object(payload),
            ]
            if let sessionID {
                params["session_id"] = .string(sessionID)
            }
            self.sendJSON([
                "jsonrpc": .string("2.0"),
                "method": .string("event"),
                "params": .object(params),
            ])
        }
    }

    func sendMalformedFrame() {
        queue.async { self.sendText("not-json") }
    }

    func disconnect() {
        queue.async {
            self.connection?.cancel()
            self.connection = nil
        }
    }

    func hasReceivedRequest(named method: String) -> Bool {
        queue.sync { receivedMethods.contains(method) }
    }

    func acceptedAuthentication() -> Bool {
        queue.sync { authenticated }
    }

    private func accept(_ connection: NWConnection) {
        self.connection?.cancel()
        self.connection = connection
        handshakeBuffer.removeAll(keepingCapacity: true)
        frameBuffer.removeAll(keepingCapacity: true)
        handshakeComplete = false
        connection.start(queue: queue)
        receiveNext()
    }

    private func receiveNext() {
        connection?.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            self.queue.async {
                guard error == nil, !isComplete else { return }
                if let data {
                    self.process(data)
                }
                self.receiveNext()
            }
        }
    }

    private func process(_ data: Data) {
        if !handshakeComplete {
            handshakeBuffer.append(data)
            guard let range = handshakeBuffer.range(of: Data("\r\n\r\n".utf8)) else { return }
            let requestData = handshakeBuffer[..<range.upperBound]
            let remainder = handshakeBuffer[range.upperBound...]
            handshakeBuffer.removeAll(keepingCapacity: true)
            guard completeHandshake(Data(requestData)) else {
                connection?.cancel()
                return
            }
            handshakeComplete = true
            frameBuffer.append(remainder)
            for event in eventsOnConnect where event == .gatewayReady {
                sendEvent(type: "gateway.ready", sessionID: nil, payload: [:])
            }
        } else {
            frameBuffer.append(data)
        }
        while let text = nextClientTextFrame() {
            handleClientText(text)
        }
    }

    private func completeHandshake(_ requestData: Data) -> Bool {
        guard let request = String(data: requestData, encoding: .utf8) else { return false }
        let lines = request.split(separator: "\r\n", omittingEmptySubsequences: false)
        let requestTarget = lines.first?.split(separator: " ").dropFirst().first
        guard let first = lines.first,
              first.hasPrefix("GET /api/ws?token="),
              requestTarget?.hasSuffix("?token=") == false,
              let key = lines.first(where: { $0.lowercased().hasPrefix("sec-websocket-key:") })
        else { return false }
        authenticated = true
        let clientKey = key.split(separator: ":", maxSplits: 1)[1].trimmingCharacters(in: .whitespaces)
        let accept = Data(Insecure.SHA1.hash(data: Data("\(clientKey)258EAFA5-E914-47DA-95CA-C5AB0DC85B11".utf8))).base64EncodedString()
        let response = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: \(accept)\r\n\r\n"
        connection?.send(content: Data(response.utf8), completion: .contentProcessed { _ in })
        return true
    }

    private func nextClientTextFrame() -> String? {
        guard frameBuffer.count >= 2 else { return nil }
        let bytes = [UInt8](frameBuffer)
        guard bytes[0] & 0x0F == 0x1 else { return nil }
        guard bytes[1] & 0x80 != 0 else { return nil }
        var offset = 2
        var length = Int(bytes[1] & 0x7F)
        if length == 126 {
            guard bytes.count >= offset + 2 else { return nil }
            length = Int(bytes[offset]) << 8 | Int(bytes[offset + 1])
            offset += 2
        } else if length == 127 {
            guard bytes.count >= offset + 8 else { return nil }
            length = bytes[offset..<(offset + 8)].reduce(0) { ($0 << 8) | Int($1) }
            offset += 8
        }
        guard bytes.count >= offset + 4 + length else { return nil }
        let mask = Array(bytes[offset..<(offset + 4)])
        offset += 4
        let payload = bytes[offset..<(offset + length)].enumerated().map { $0.element ^ mask[$0.offset % 4] }
        frameBuffer.removeFirst(offset + length)
        return String(bytes: payload, encoding: .utf8)
    }

    private func handleClientText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let request = try? JSONDecoder().decode([String: JSONValue].self, from: data),
              let method = request["method"]?.stringValue,
              let id = request["id"]
        else { return }
        receivedMethods.append(method)
        if rpcErrors.contains(method) {
            sendJSON([
                "jsonrpc": .string("2.0"),
                "id": id,
                "error": .object(["message": .string("fixture failure")]),
            ])
        } else if let result = responses[method] {
            sendJSON([
                "jsonrpc": .string("2.0"),
                "id": id,
                "result": result,
            ])
        }
    }

    private func sendJSON(_ object: [String: JSONValue]) {
        guard let data = try? JSONEncoder().encode(object),
              let text = String(data: data, encoding: .utf8)
        else { return }
        sendText(text)
    }

    private func sendText(_ text: String) {
        let payload = [UInt8](text.utf8)
        var frame: [UInt8] = [0x81]
        if payload.count < 126 {
            frame.append(UInt8(payload.count))
        } else {
            frame += [126, UInt8((payload.count >> 8) & 0xFF), UInt8(payload.count & 0xFF)]
        }
        frame += payload
        connection?.send(content: Data(frame), completion: .contentProcessed { _ in })
    }

    private enum FixtureError: Error {
        case startFailed
    }
}

private final class FixtureStartupSignal: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var didFail = false

    var failed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return didFail
    }

    func finish(failed: Bool) {
        lock.lock()
        didFail = failed
        lock.unlock()
        semaphore.signal()
    }

    func wait(timeout: DispatchTime) -> Bool {
        semaphore.wait(timeout: timeout) == .success
    }
}
