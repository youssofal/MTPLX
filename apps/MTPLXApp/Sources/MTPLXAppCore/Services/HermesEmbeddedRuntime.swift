import Foundation

public enum HermesProfileRoutingState: Equatable, Sendable {
    case mtplx
    case external
    case unavailable(String)
}

public struct HermesServeLaunchSpec: Equatable, Sendable {
    public let executableURL: URL
    public let arguments: [String]
    public let environment: [String: String]
    public let token: String
    public let launchID: String
    public let parentPID: Int32

    public init(
        executableURL: URL,
        arguments: [String],
        environment: [String: String],
        token: String,
        launchID: String,
        parentPID: Int32
    ) {
        self.executableURL = executableURL
        self.arguments = arguments
        self.environment = environment
        self.token = token
        self.launchID = launchID
        self.parentPID = parentPID
    }
}

/// The lifecycle operations provided by the embedded-runtime implementation.
/// Task 1 deliberately declares this contract without launching a process.
public protocol HermesSidecarControlling: Sendable {
    var processIdentifier: Int32 { get }
    func stop()
}

public enum HermesSessionOwnership: Equatable, Sendable {
    case appOwned
    case external
    case unavailable(String)
}

public protocol HermesEmbeddedRuntime: Sendable {
    func routingState(
        for profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) -> HermesProfileRoutingState
    func startEmbeddedSidecar(
        profile: HermesProfile,
        configuration: MTPLXAppConfiguration
    ) async throws -> any HermesSidecarControlling
    func sessionOwnership(
        profile: HermesProfile,
        sessionID: String,
        ownedSidecarPID: Int32?
    ) -> HermesSessionOwnership
    @discardableResult func reapOrphanedEmbeddedSidecars() -> [Int32]
}
