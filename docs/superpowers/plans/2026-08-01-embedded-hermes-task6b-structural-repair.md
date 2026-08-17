# Embedded Hermes Task 6B: Structural Approval Recovery Repair

**Goal:** Preserve a visible, fail-closed Hermes approval FIFO across transport-loss and overflow ordering, while keeping ordinary disconnect cleanup unchanged.

**Authorized context:** This addendum follows the Task 6 five-round breaker. The user explicitly authorized the recommended structural replan. It does not broaden the feature beyond the approved embedded Hermes design.

**Constraints:**

- Keep Hermes' native session-scoped, head-only approval FIFO contract.
- Never retry an approval whose delivery is ambiguous.
- Tear down only the currently owned embedded client and sidecar, exactly once.
- Preserve the blocked inbox head until a successful fresh sidecar/client handshake.
- Ordinary disconnects without an in-flight approval response continue to clear transient pending state.
- No GUI, profile, registry, root-gateway, Telegram, or native Hermes changes.

### Task 1: Separate Transport Teardown from Approval Recovery State

**Files:**

- Modify: `apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift`
- Modify: `apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift`

**Step 1: Add failing ordering regressions**

- Model the real client order: fail pending approval RPC continuations, then invoke `onDisconnect` in the same transport termination turn.
- Assert suspended approval A plus later B remain in the inbox, A is visibly projected, the pipeline is blocked, no retry occurs, and the owned client/sidecar are closed/stopped exactly once.
- Assert the old RPC completion cannot clear or advance the retained head.
- Assert a reconnect that has not reached `gateway.ready` preserves blocked A; only a successful fresh sidecar/client handshake clears it.
- In auto mode, suspend A and overflow the unified 64-entry inbox. Assert A is projected before reset, the pipeline is blocked, no additional RPC is sent, and teardown is exact-once.
- Preserve ordinary disconnect coverage for clarification/sudo/secret with no in-flight approval response.

**Step 2: Implement transport-only ownership release**

- Split gateway detachment/close/owned-sidecar stop from pending-request lifecycle disposal.
- Before an owned disconnect releases transport, inspect the current response lease. If it is the matching approval inbox head, atomically retain/project the head, clear only the transient lease, mark the pipeline blocked, add a generic error, and then release transport.
- For ordinary disconnects, explicitly dispose the pending-request lifecycle before transport release.
- Ensure the later approval task catch is harmless and cannot retry, advance, clear, or stop again.

**Step 3: Unify ambiguous-response and overflow blocking**

- Factor a helper that marks the pipeline blocked, projects the retained inbox head, clears the matching transient response lease, and emits only a generic message.
- Call it before transport reset for both ambiguous response failures and inbox overflow.
- Keep inbox, fingerprints, and blocked state across failed/pre-ready reconnects.
- Clear them only after a successful fresh sidecar/client handshake or an explicit profile/session/stop lifecycle boundary.

**Step 4: Verify**

Run from `apps/MTPLXApp` with:

```bash
DEVELOPER_DIR=/Volumes/nugly/Applications/Xcode-beta.app/Contents/Developer \
COPYFILE_DISABLE=1 swift test \
  --scratch-path /tmp/mtplx-embedded-hermes-task6b-swiftpm \
  --filter HermesAgentStoreTests

DEVELOPER_DIR=/Volumes/nugly/Applications/Xcode-beta.app/Contents/Developer \
COPYFILE_DISABLE=1 swift test \
  --scratch-path /tmp/mtplx-embedded-hermes-task6b-swiftpm \
  --filter HermesGatewayClientTests

DEVELOPER_DIR=/Volumes/nugly/Applications/Xcode-beta.app/Contents/Developer \
COPYFILE_DISABLE=1 swift test \
  --scratch-path /tmp/mtplx-embedded-hermes-task6b-swiftpm --quiet
```

Expected: focused and full suites pass, `git diff --check` is clean, and no new warnings originate from the changed files.

**Step 5: Commit**

```bash
git add apps/MTPLXApp/Sources/MTPLXAppCore/Stores/HermesAgentStore.swift \
  apps/MTPLXApp/Tests/MTPLXAppCoreTests/HermesAgentStoreTests.swift
git commit -m "fix(hermes): preserve blocked approval recovery"
```
