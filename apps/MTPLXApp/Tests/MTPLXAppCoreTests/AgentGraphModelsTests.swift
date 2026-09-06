import Foundation
import XCTest
@testable import MTPLXAppCore

final class AgentGraphModelsTests: XCTestCase {
    private final class StubURLProtocol: URLProtocol {
        nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

        override class func canInit(with request: URLRequest) -> Bool { true }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            guard let handler = Self.handler else {
                client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
                return
            }
            do {
                let (response, data) = try handler(request)
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }

        override func stopLoading() {}
    }

    private final class RequestRecorder: @unchecked Sendable {
        struct Request {
            let method: String?
            let path: String?
            let body: Data?
        }

        private let lock = NSLock()
        private var storage: [Request] = []

        func append(_ request: URLRequest) {
            let body = request.httpBody ?? request.httpBodyStream.flatMap(Self.readAll)
            lock.lock()
            storage.append(
                Request(
                    method: request.httpMethod,
                    path: request.url?.path,
                    body: body
                )
            )
            lock.unlock()
        }

        func snapshot() -> [Request] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }

        private static func readAll(_ stream: InputStream) -> Data? {
            stream.open()
            defer { stream.close() }
            var output = Data()
            var buffer = [UInt8](repeating: 0, count: 4_096)
            while stream.hasBytesAvailable {
                let count = stream.read(&buffer, maxLength: buffer.count)
                if count < 0 {
                    return nil
                }
                if count == 0 {
                    break
                }
                output.append(buffer, count: count)
            }
            return output
        }
    }

    override func tearDown() {
        StubURLProtocol.handler = nil
        super.tearDown()
    }

    func testGraphDefinitionDecodesLoopNodeAndPinnedRevisionMetadata() throws {
        let json = #"""
        {
          "graphs": [{
            "id": "coding-graph",
            "project_id": "project",
            "workspace_id": "workspace",
            "name": "Coding Graph",
            "description": "",
            "schema_version": 1,
            "revision": 3,
            "inputs": {},
            "outputs": {},
            "nodes": [{
              "id": "repeat",
              "type": "loop",
              "name": "Verification Loop",
              "config": {"max_iterations": 4},
              "timeout_seconds": null,
              "retry": {},
              "approval": {}
            }],
            "edges": [],
            "limits": {"max_steps": 20},
            "policies": {"write": "ask"},
            "runtime_requirements": {"backend": "mtplx"},
            "retry": {},
            "timeout_seconds": 3600,
            "approval_requirements": {},
            "created_at": "2026-08-29T18:00:00Z",
            "updated_at": "2026-08-29T18:01:00.123456Z",
            "content_sha256": "abc123"
          }]
        }
        """#
        let payload = try MTPLXAPIClient.makeDefaultDecoder().decode(
            AgentGraphsPayload.self,
            from: Data(json.utf8)
        )
        let graph = try XCTUnwrap(payload.graphs.first)
        XCTAssertEqual(graph.schemaVersion, 1)
        XCTAssertEqual(graph.revision, 3)
        XCTAssertEqual(graph.contentSHA256, "abc123")
        XCTAssertEqual(graph.nodes.first?.type, "loop")
        XCTAssertEqual(graph.nodes.first?.config.values["max_iterations"]?.intValue, 4)
    }

    func testGraphRunDecodesDurableNodeAndLoopCheckpointState() throws {
        let json = #"""
        {
          "id": "graph-run-1",
          "graph_id": "coding-graph",
          "graph_revision": 3,
          "graph_sha256": "abc123",
          "workspace_id": "workspace",
          "project_id": "project",
          "workspace_root": "/tmp/project",
          "status": "waiting_approval",
          "pinned_model": "local-model",
          "runtime_profile": "balanced",
          "inputs": {"goal": "verify"},
          "outputs": {},
          "node_states": {
            "repeat": {
              "status": "waiting_approval",
              "iterations_completed": 2,
              "active_iteration": 3,
              "loop_outputs": ["one", "two"]
            }
          },
          "current_node_id": "repeat",
          "pending_approval_id": "approval-1",
          "resource_metrics": {"steps_completed": 2},
          "created_at": "2026-08-29T18:00:00Z",
          "updated_at": "2026-08-29T18:01:00Z",
          "state_version": 9,
          "pause_requested": false,
          "error": null
        }
        """#
        let run = try MTPLXAPIClient.makeDefaultDecoder().decode(
            AgentGraphRun.self,
            from: Data(json.utf8)
        )
        XCTAssertEqual(run.graphRevision, 3)
        XCTAssertEqual(run.workspaceRoot, "/tmp/project")
        XCTAssertEqual(run.pinnedModel, "local-model")
        XCTAssertEqual(run.pendingApprovalID, "approval-1")
        XCTAssertEqual(
            run.nodeStates["repeat"]?.values["iterations_completed"]?.intValue,
            2
        )
        XCTAssertEqual(run.resourceMetrics.values["steps_completed"]?.intValue, 2)
    }

    func testDelegationDecodesDurableBudgetUsageAndAttempts() throws {
        let json = #"""
        {
          "id": "delegation-1",
          "workspace_id": "workspace",
          "parent_run_id": "parent-run",
          "child_run_id": "child-run",
          "role": "reviewer",
          "permissions": ["read", "search"],
          "prompt": "Review it",
          "model": "local-model",
          "budget": 512,
          "context_window": 65536,
          "profile_sha256": "profile-sha",
          "status": "running",
          "created_at": "2026-08-29T18:00:00Z",
          "updated_at": "2026-08-29T18:01:00Z",
          "worktree_path": "/tmp/worktree",
          "worktree_commit": "abc123",
          "source_delegation_id": null,
          "tokens_used": 300,
          "attempts": 2,
          "evidence": null,
          "error": null
        }
        """#
        let delegation = try MTPLXAPIClient.makeDefaultDecoder().decode(
            AgentDelegation.self,
            from: Data(json.utf8)
        )
        XCTAssertEqual(delegation.tokensUsed, 300)
        XCTAssertEqual(delegation.remainingTokenBudget, 212)
        XCTAssertEqual(delegation.attempts, 2)
    }

    func testGraphApprovalRequestUsesExactAPIFieldNames() throws {
        let request = AgentGraphApprovalRequest(
            approvalID: "approval-1",
            decision: "approved",
            resolvedBy: "desktop",
            reason: "reviewed"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        XCTAssertEqual(object["approval_id"] as? String, "approval-1")
        XCTAssertEqual(object["resolved_by"] as? String, "desktop")
        XCTAssertEqual(object["resume"] as? Bool, true)
    }

    func testGraphRunApprovalsUsesRunScopedPendingEndpoint() async throws {
        StubURLProtocol.handler = { request in
            XCTAssertEqual(request.httpMethod, "GET")
            XCTAssertEqual(request.url?.path, "/v1/mtplx/graph-runs/run-1/approvals")
            let query = URLComponents(url: try XCTUnwrap(request.url), resolvingAgainstBaseURL: false)
            XCTAssertEqual(
                Dictionary(uniqueKeysWithValues: query?.queryItems?.compactMap { item in
                    item.value.map { (item.name, $0) }
                } ?? []),
                ["status": "pending", "limit": "100"]
            )
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (
                response,
                Data(
                    #"""
                    {
                      "run_id": "run-1",
                      "approvals": [{
                        "id": "approval-1",
                        "workspace_id": "workspace",
                        "run_id": "run-1",
                        "tool": "shell",
                        "action": "execute",
                        "description": "Run the verification command",
                        "target": "swift test",
                        "risk": "medium",
                        "status": "pending",
                        "created_at": "2026-08-29T18:00:00Z",
                        "arguments": {"command": "swift test"},
                        "arguments_sha256": "abc123",
                        "expires_at": "2026-08-29T18:10:00Z",
                        "resolved_at": null,
                        "resolved_by": null,
                        "reason": null,
                        "consumed_at": null,
                        "consumed_by": null
                      }]
                    }
                    """#.utf8
                )
            )
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let client = MTPLXAPIClient(
            baseURL: URL(string: "http://127.0.0.1:9")!,
            session: URLSession(configuration: configuration)
        )

        let payload = try await client.graphRunApprovals(runID: "run-1")

        XCTAssertEqual(payload.runID, "run-1")
        XCTAssertEqual(payload.approvals.map(\.id), ["approval-1"])
        XCTAssertEqual(payload.approvals.first?.target, "swift test")
    }

    func testGraphLifecycleRequestsUseExactEndpointsAndBodies() async throws {
        let recorder = RequestRecorder()
        StubURLProtocol.handler = { request in
            recorder.append(request)
            let path = try XCTUnwrap(request.url?.path)
            let status: String
            switch path {
            case "/v1/mtplx/graph-runs":
                status = "running"
            case "/v1/mtplx/graph-runs/run-1/pause":
                status = "paused"
            case "/v1/mtplx/graph-runs/run-1/resume":
                status = "running"
            case "/v1/mtplx/graph-runs/run-1/cancel":
                status = "cancelled"
            case "/v1/mtplx/graph-runs/run-1/retry":
                status = "queued"
            default:
                throw MTPLXAPIClientError.httpStatus(404, path)
            }
            let response = try XCTUnwrap(
                HTTPURLResponse(
                    url: request.url!,
                    statusCode: 200,
                    httpVersion: "HTTP/1.1",
                    headerFields: ["Content-Type": "application/json"]
                )
            )
            return (response, Self.graphRunData(status: status))
        }

        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        let client = MTPLXAPIClient(
            baseURL: URL(string: "http://127.0.0.1:9")!,
            session: URLSession(configuration: configuration)
        )

        let started = try await client.startGraphRun(
            graphID: "coding-graph",
            revision: 7,
            inputs: DynamicObject(values: ["goal": .string("ship")]),
            model: "local-model",
            runtimeProfile: "balanced",
            runID: "run-1"
        )
        let paused = try await client.pauseGraphRun(runID: "run-1")
        let resumed = try await client.resumeGraphRun(runID: "run-1")
        let cancelled = try await client.cancelGraphRun(runID: "run-1")
        let retried = try await client.retryGraphRun(
            runID: "run-1",
            nodeID: "verify-loop",
            allowSideEffectRetry: true,
            forceNewSideEffect: true
        )

        XCTAssertEqual(
            [started.status, paused.status, resumed.status, cancelled.status, retried.status],
            ["running", "paused", "running", "cancelled", "queued"]
        )

        let requests = recorder.snapshot()
        XCTAssertEqual(requests.count, 5)
        XCTAssertEqual(requests.map(\.method), Array(repeating: "POST", count: 5))
        XCTAssertEqual(
            requests.compactMap(\.path),
            [
                "/v1/mtplx/graph-runs",
                "/v1/mtplx/graph-runs/run-1/pause",
                "/v1/mtplx/graph-runs/run-1/resume",
                "/v1/mtplx/graph-runs/run-1/cancel",
                "/v1/mtplx/graph-runs/run-1/retry",
            ]
        )

        let startBody = try Self.jsonObject(requests[0])
        XCTAssertEqual(startBody["graph_id"] as? String, "coding-graph")
        XCTAssertEqual(startBody["revision"] as? Int, 7)
        XCTAssertEqual((startBody["inputs"] as? [String: Any])?["goal"] as? String, "ship")
        XCTAssertEqual(startBody["model"] as? String, "local-model")
        XCTAssertEqual(startBody["runtime_profile"] as? String, "balanced")
        XCTAssertEqual(startBody["run_id"] as? String, "run-1")
        XCTAssertEqual(startBody["start"] as? Bool, true)

        for request in requests[1...3] {
            XCTAssertEqual(try Self.jsonObject(request).count, 0)
        }

        let retryBody = try Self.jsonObject(requests[4])
        XCTAssertEqual(retryBody["node_id"] as? String, "verify-loop")
        XCTAssertEqual(retryBody["allow_side_effect_retry"] as? Bool, true)
        XCTAssertEqual(retryBody["force_new_side_effect"] as? Bool, true)
    }

    private static func jsonObject(_ request: RequestRecorder.Request) throws -> [String: Any] {
        let data = try XCTUnwrap(request.body)
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }

    private static func graphRunData(status: String) -> Data {
        Data(
            """
            {
              "id": "run-1",
              "graph_id": "coding-graph",
              "graph_revision": 7,
              "graph_sha256": "abc123",
              "workspace_id": "workspace",
              "project_id": "workspace",
              "workspace_root": "/tmp/project",
              "status": "\(status)",
              "pinned_model": "local-model",
              "runtime_profile": "balanced",
              "inputs": {"goal": "ship"},
              "outputs": {},
              "node_states": {},
              "current_node_id": "verify-loop",
              "pending_approval_id": null,
              "resource_metrics": {},
              "created_at": "2026-08-29T18:00:00Z",
              "updated_at": "2026-08-29T18:01:00Z",
              "state_version": 9,
              "pause_requested": false,
              "error": null
            }
            """.utf8
        )
    }
}
