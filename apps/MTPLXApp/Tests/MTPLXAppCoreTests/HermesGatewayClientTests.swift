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
        await eventually { backend.peerWasDisconnected() }
        client.close()
    }

    func testUnknownResponseIDTerminatesPendingRequest() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        defer { client.close() }
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let completed = expectation(description: "pending request fails")
        var observed: Error?
        Task {
            do {
                _ = try await client.call(method: "wrong-id", params: [:])
            } catch {
                observed = error
                completed.fulfill()
            }
        }

        await eventually { backend.hasReceivedRequest(named: "wrong-id") }
        backend.sendResult(id: 99)
        await fulfillment(of: [completed], timeout: 0.5)
        XCTAssertEqual(observed as? HermesGatewayClientError, .malformedResponse)
        await eventually { backend.peerWasDisconnected() }
    }

    func testLateDuplicateOfRetiredIDDoesNotTerminateNewPendingRequest() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        defer { client.close() }
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)

        let first = Task { try await client.call(method: "first", params: [:]) }
        await eventually { backend.hasReceivedRequest(named: "first") }
        backend.sendResult(id: 1, result: .string("first"))
        let firstValue = try await first.value
        XCTAssertEqual(firstValue, .string("first"))

        let second = Task { try await client.call(method: "second", params: [:]) }
        await eventually { backend.hasReceivedRequest(named: "second") }
        backend.sendResult(id: 1, result: .string("late duplicate"))
        try await Task.sleep(for: .milliseconds(25))
        backend.sendResult(id: 2, result: .string("second"))
        let secondValue = try await second.value
        XCTAssertEqual(secondValue, .string("second"))
    }

    func testLateDuplicateOfFailedRetiredIDDoesNotTerminateNewPendingRequest() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        defer { client.close() }
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)

        let first = Task { try await client.call(method: "first-error", params: [:]) }
        await eventually { backend.hasReceivedRequest(named: "first-error") }
        backend.sendRPCError(id: 1)
        do {
            _ = try await first.value
            XCTFail("Expected RPC error")
        } catch {
            XCTAssertEqual(error as? HermesGatewayClientError, .rpcError)
        }

        let second = Task { try await client.call(method: "second-after-error", params: [:]) }
        await eventually { backend.hasReceivedRequest(named: "second-after-error") }
        backend.sendRPCError(id: 1)
        try await Task.sleep(for: .milliseconds(25))
        backend.sendResult(id: 2, result: .string("second"))
        let secondValue = try await second.value
        XCTAssertEqual(secondValue, .string("second"))
    }

    func testUnknownResponseIDTerminatesAllConcurrentPendingRequests() async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        defer { client.close() }
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let completed = expectation(description: "both pending requests fail")
        completed.expectedFulfillmentCount = 2
        var observed: [HermesGatewayClientError] = []
        Task {
            do {
                _ = try await client.call(method: "concurrent-a", params: [:])
            } catch let error as HermesGatewayClientError {
                observed.append(error)
                completed.fulfill()
            } catch {
                XCTFail("Unexpected error type")
            }
        }
        Task {
            do {
                _ = try await client.call(method: "concurrent-b", params: [:])
            } catch let error as HermesGatewayClientError {
                observed.append(error)
                completed.fulfill()
            } catch {
                XCTFail("Unexpected error type")
            }
        }

        await eventually {
            backend.hasReceivedRequest(named: "concurrent-a") &&
                backend.hasReceivedRequest(named: "concurrent-b")
        }
        backend.sendResult(id: 99)
        await fulfillment(of: [completed], timeout: 0.5)
        XCTAssertEqual(observed, [.malformedResponse, .malformedResponse])
    }

    func testUnsupportedResponseIDsFailClosedWithoutTrapping() async throws {
        for responseID: JSONValue in [
            .number(-1),
            .number(1.5),
            .number(9_007_199_254_740_992),
            .number(9_223_372_036_854_775_808),
        ] {
            try await assertMalformedResponseID(responseID)
        }
    }

    func testRequestIDExhaustionFailsClosedWithoutReusingID() async throws {
        let maxSafeID = 9_007_199_254_740_991
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(
            url: backend.webSocketURL(token: UUID().uuidString),
            startingRequestID: maxSafeID
        )
        defer { client.close() }
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)

        let finalRequest = Task { try await client.call(method: "last-safe-id", params: [:]) }
        await eventually { backend.hasReceivedRequest(named: "last-safe-id") }
        backend.sendResult(id: maxSafeID, result: .string("final"))
        let finalValue = try await finalRequest.value
        XCTAssertEqual(finalValue, .string("final"))

        let completed = expectation(description: "request ID exhaustion")
        var observed: Error?
        Task {
            do {
                _ = try await client.call(method: "after-id-exhaustion", params: [:])
            } catch {
                observed = error
                completed.fulfill()
            }
        }
        await fulfillment(of: [completed], timeout: 0.5)
        XCTAssertEqual(observed as? HermesGatewayClientError, .requestIDExhausted)
        XCTAssertFalse(backend.hasReceivedRequest(named: "after-id-exhaustion"))
        await eventually { backend.peerWasDisconnected() }
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

    private func assertMalformedResponseID(_ responseID: JSONValue) async throws {
        let backend = try FakeHermesGateway(eventsOnConnect: [.gatewayReady])
        let client = URLSessionHermesGatewayClient(url: backend.webSocketURL(token: UUID().uuidString))
        defer { client.close() }
        try await client.connectAndWaitUntilReady(timeoutSeconds: 1)
        let completed = expectation(description: "invalid ID fails pending request")
        var observed: Error?
        Task {
            do {
                _ = try await client.call(method: "invalid-id", params: [:])
            } catch {
                observed = error
                completed.fulfill()
            }
        }
        await eventually { backend.hasReceivedRequest(named: "invalid-id") }
        backend.sendResult(id: responseID)
        await fulfillment(of: [completed], timeout: 0.5)
        XCTAssertEqual(observed as? HermesGatewayClientError, .malformedResponse)
        await eventually { backend.peerWasDisconnected() }
    }
}
