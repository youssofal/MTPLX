import Foundation
import XCTest
@testable import MTPLXAppCore

@MainActor
final class HermesGatewayClientTests: XCTestCase {
    func testConnectWaitsForGatewayReadyBeforeReturning() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        let connect = Task { try await client.connectAndWaitUntilReady(timeoutSeconds: 1) }

        await eventually { backend.acceptedAuthentication() }
        XCTAssertFalse(connect.isCancelled)
        backend.sendEvent(type: "gateway.ready", sessionID: nil, payload: [:])
        try await connect.value
        XCTAssertTrue(backend.acceptedAuthentication())
        client.close()
    }

    func testRPCResponseAndEventAreDecoded() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        backend.respond(to: "session.list", result: .object(["sessions": .array([])]))
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        var received: [HermesGatewayEvent] = []
        client.onEvent = { received.append($0) }

        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let value = try await client.call(method: "session.list", params: ["limit": .number(200)])
        guard case .object(let response) = value,
              case .array(let sessions)? = response["sessions"]
        else { return XCTFail("Expected session array") }
        XCTAssertEqual(sessions, [])
        backend.sendEvent(type: "message.delta", sessionID: "live-1", payload: ["text": .string("Hi")])
        await eventually { received.contains(where: { $0.type == "message.delta" }) }
        client.close()
    }

    func testReadinessTimesOutWhenReadyEventNeverArrives() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))

        await XCTAssertThrowsErrorAsync(try await client.connectAndWaitUntilReady(timeoutSeconds: 0.05)) { error in
            guard case HermesGatewayClientError.readinessTimedOut = error else {
                return XCTFail("Expected readiness timeout")
            }
        }
        client.close()
    }

    func testNonReadyEventDoesNotSatisfyReadinessGate() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        let connect = Task { try await client.connectAndWaitUntilReady(timeoutSeconds: 0.05) }

        await eventually { backend.acceptedAuthentication() }
        backend.sendEvent(type: "message.delta", sessionID: "live-1", payload: [:])
        await XCTAssertThrowsErrorAsync(try await connect.value) { error in
            guard case HermesGatewayClientError.readinessTimedOut = error else {
                return XCTFail("Expected readiness timeout")
            }
        }
        client.close()
    }

    func testRPCErrorFailsRequestWithoutLeakingRemoteDescription() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        backend.respondWithRPCError(to: "session.list")
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)

        await XCTAssertThrowsErrorAsync(try await client.call(method: "session.list", params: [:])) { error in
            guard case HermesGatewayClientError.rpcError = error else {
                return XCTFail("Expected RPC error")
            }
        }
        client.close()
    }

    func testDisconnectFailsPendingRequest() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let request = Task { try await client.call(method: "session.list", params: [:]) }

        await eventually { backend.hasReceivedRequest(named: "session.list") }
        backend.disconnect()
        await XCTAssertThrowsErrorAsync(try await request.value) { error in
            guard case HermesGatewayClientError.disconnected = error else {
                return XCTFail("Expected disconnect")
            }
        }
        client.close()
    }

    func testDisconnectFailsReadinessWaiter() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        let connect = Task { try await client.connectAndWaitUntilReady(timeoutSeconds: 1) }

        await eventually { backend.acceptedAuthentication() }
        backend.disconnect()
        await XCTAssertThrowsErrorAsync(try await connect.value) { error in
            guard case HermesGatewayClientError.disconnected = error else {
                return XCTFail("Expected disconnect")
            }
        }
        client.close()
    }

    func testCloseFailsPendingRequest() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let request = Task { try await client.call(method: "session.list", params: [:]) }

        await eventually { backend.hasReceivedRequest(named: "session.list") }
        client.close()
        await XCTAssertThrowsErrorAsync(try await request.value) { error in
            guard case HermesGatewayClientError.disconnected = error else {
                return XCTFail("Expected close failure")
            }
        }
    }

    func testMalformedFrameFailsPendingRequest() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let request = Task { try await client.call(method: "session.list", params: [:]) }

        await eventually { backend.hasReceivedRequest(named: "session.list") }
        backend.sendMalformedFrame()
        await XCTAssertThrowsErrorAsync(try await request.value) { error in
            guard case HermesGatewayClientError.malformedResponse = error else {
                return XCTFail("Expected malformed response")
            }
        }
        client.close()
    }

    func testAuthenticatedQueryIsSentWithoutSurfacingItInErrors() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        XCTAssertTrue(backend.acceptedAuthentication())
        client.close()
    }

    private func eventually(
        timeout: Duration = .seconds(1),
        _ condition: @escaping @MainActor () -> Bool
    ) async {
        let clock = ContinuousClock()
        let deadline = clock.now + timeout
        while !condition(), clock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(condition())
    }

    private func XCTAssertThrowsErrorAsync<T>(
        _ expression: @autoclosure () async throws -> T,
        _ handler: (Error) -> Void
    ) async {
        do {
            _ = try await expression()
            XCTFail("Expected an error")
        } catch {
            handler(error)
        }
    }
}
