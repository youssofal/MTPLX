public enum HermesComposerPolicy {
    /// The composer may accept text for an existing writable session or for a
    /// selected profile that can create a fresh session on first send.
    public static func acceptsInput(
        gatewayReady: Bool,
        hasPendingRequest: Bool,
        activeSessionWritable: Bool,
        hasActiveSession: Bool,
        hasAvailableProfile: Bool
    ) -> Bool {
        guard gatewayReady, !hasPendingRequest else { return false }
        if activeSessionWritable { return true }
        return !hasActiveSession && hasAvailableProfile
    }
}
