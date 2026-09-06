import Foundation

/// Internal workspace-tool schema and reference executor.
///
/// Every path is resolved below the selected workspace root. Mutating tools
/// are never exposed through the public chat dispatch path. Production chat
/// execution goes through the daemon API, which binds exact approvals and
/// durable events directly to the execution boundary.
struct MTPLXWorkspaceToolService: Sendable {
    private static let maxReadBytes = 2 * 1024 * 1024
    private static let maxReadCharacters = 40_000
    private static let maxSearchBytes = 1 * 1024 * 1024
    private static let maxResults = 100
    private static let maxWriteCharacters = 1_000_000
    private static let excludedDirectoryNames: Set<String> = [
        ".git", ".build", ".swiftpm", "DerivedData", "node_modules",
        ".venv", "venv", "__pycache__", ".mtplx"
    ]
    private static let excludedFileNames: Set<String> = [
        ".env", ".env.local", ".env.production"
    ]

    init() {}

    func toolDefinitions() -> [ChatRequestTool] {
        [
            definition(
                name: "list_files",
                description: "List files and directories inside the selected MTPLX workspace.",
                properties: [
                    "path": stringProperty("Relative directory path. Defaults to the workspace root."),
                    "depth": integerProperty("Maximum directory depth, from 0 to 4.")
                ],
                required: []
            ),
            definition(
                name: "read_file",
                description: "Read a UTF-8 text file inside the selected MTPLX workspace.",
                properties: [
                    "path": stringProperty("Relative file path inside the workspace."),
                    "max_chars": integerProperty("Maximum characters to return, capped at 40000.")
                ],
                required: ["path"]
            ),
            definition(
                name: "search_files",
                description: "Search text files inside the selected MTPLX workspace and return matching paths, line numbers, and snippets.",
                properties: [
                    "query": stringProperty("Text to find."),
                    "path": stringProperty("Optional relative directory to search."),
                    "case_sensitive": booleanProperty("Whether matching should be case-sensitive."),
                    "max_results": integerProperty("Maximum matches, capped at 100.")
                ],
                required: ["query"]
            ),
            definition(
                name: "inspect_repo",
                description: "Inspect the selected repository's branch, project markers, and top-level layout without changing files.",
                properties: [:],
                required: []
            ),
            definition(
                name: "git_status",
                description: "Read the selected repository's current branch and changed-file status.",
                properties: [:],
                required: []
            ),
            definition(
                name: "git_diff",
                description: "Read the selected repository's working-tree or staged diff.",
                properties: [
                    "scope": stringProperty("unstaged, staged, or both."),
                    "path": stringProperty("Optional relative path to limit the diff.")
                ],
                required: []
            ),
            definition(
                name: "write_file",
                description: "Write UTF-8 text to a file inside the selected MTPLX workspace. MTPLX asks the user for approval before this tool runs.",
                properties: [
                    "path": stringProperty("Relative file path inside the workspace."),
                    "content": stringProperty("Complete replacement file content.")
                ],
                required: ["path", "content"]
            ),
            definition(
                name: "apply_patch",
                description: "Apply a unified Git patch inside the selected MTPLX workspace. MTPLX validates the patch and asks the user for approval before applying it.",
                properties: [
                    "patch": stringProperty("Unified diff to validate and apply.")
                ],
                required: ["patch"]
            ),
            definition(
                name: "run_tests",
                description: "Run the repository's test command inside the selected MTPLX workspace. MTPLX asks the user for approval before execution.",
                properties: [
                    "command": stringProperty("Optional test command. If omitted, MTPLX detects pytest, Swift Package Manager, or npm.")
                ],
                required: []
            ),
            definition(
                name: "run_command",
                description: "Run a shell command with the selected MTPLX workspace as its working directory. MTPLX asks the user for approval before this tool runs.",
                properties: [
                    "command": stringProperty("Command to run."),
                    "timeout_seconds": integerProperty("Timeout from 1 to 60 seconds.")
                ],
                required: ["command"]
            )
        ]
    }

    func dispatch(
        name: String,
        argumentsJSON: String,
        workspaceRoot: String,
        approved: Bool
    ) async -> String {
        guard let root = resolvedRoot(workspaceRoot) else {
            return jsonObject([
                "error": "invalid_workspace",
                "note": "The selected workspace root does not exist or is not a directory."
            ])
        }
        let arguments = parseObject(argumentsJSON)
        switch name {
        case "list_files":
            return listFiles(arguments, root: root)
        case "read_file":
            return readFile(arguments, root: root)
        case "search_files":
            return searchFiles(arguments, root: root)
        case "inspect_repo":
            return await inspectRepo(root: root)
        case "git_status":
            return await gitStatus(root: root)
        case "git_diff":
            return await gitDiff(arguments, root: root)
        case "write_file":
            guard approved else { return approvalRequired(name: name) }
            return writeFile(arguments, root: root)
        case "apply_patch":
            guard approved else { return approvalRequired(name: name) }
            return await applyPatch(arguments, root: root)
        case "run_tests":
            guard approved else { return approvalRequired(name: name) }
            return await runTests(arguments, root: root)
        case "run_command":
            guard approved else { return approvalRequired(name: name) }
            return await runCommand(arguments, root: root)
        default:
            return jsonObject([
                "error": "unknown_workspace_tool",
                "tool": name
            ])
        }
    }

    private func resolvedRoot(_ path: String) -> URL? {
        let root = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else { return nil }
        return root
    }

    private func resolvedPath(_ rawPath: String, root: URL) -> URL? {
        let relative = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidate = root
            .appendingPathComponent(relative.isEmpty ? "." : relative)
            .resolvingSymlinksInPath()
            .standardizedFileURL
        guard candidate.path == root.path || candidate.path.hasPrefix(root.path + "/") else {
            return nil
        }
        return candidate
    }

    private func listFiles(_ arguments: [String: Any], root: URL) -> String {
        let rawPath = string(arguments, key: "path")
        guard let directory = resolvedPath(rawPath, root: root) else {
            return jsonObject(["error": "path_outside_workspace", "path": rawPath])
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return jsonObject(["error": "not_a_directory", "path": rawPath])
        }
        let maxDepth = min(max(integer(arguments, key: "depth", default: 2), 0), 4)
        var files: [String] = []
        var directories: [String] = []
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
            options: [.skipsPackageDescendants]
        ) else {
            return jsonObject(["error": "directory_unreadable", "path": rawPath])
        }
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            if Self.excludedDirectoryNames.contains(name) {
                enumerator.skipDescendants()
                continue
            }
            if Self.excludedFileNames.contains(name) { continue }
            let relative = relativePath(url, from: root)
            let depth = relative.split(separator: "/").count - 1
            if depth > maxDepth {
                enumerator.skipDescendants()
                continue
            }
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            if values?.isSymbolicLink == true {
                enumerator.skipDescendants()
                continue
            }
            if values?.isDirectory == true {
                directories.append(relative)
            } else {
                files.append(relative)
            }
            if files.count + directories.count >= 500 { break }
        }
        return jsonObject([
            "path": rawPath.isEmpty ? "." : rawPath,
            "files": files.sorted(),
            "directories": directories.sorted(),
            "truncated": files.count + directories.count >= 500
        ])
    }

    private func readFile(_ arguments: [String: Any], root: URL) -> String {
        let rawPath = string(arguments, key: "path")
        guard let file = resolvedPath(rawPath, root: root) else {
            return jsonObject(["error": "path_outside_workspace", "path": rawPath])
        }
        guard !Self.excludedFileNames.contains(file.lastPathComponent) else {
            return jsonObject(["error": "sensitive_file_blocked", "path": rawPath])
        }
        guard let values = try? file.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey]),
              values.isDirectory != true
        else {
            return jsonObject(["error": "not_a_file", "path": rawPath])
        }
        guard let size = values.fileSize, size <= Self.maxReadBytes else {
            return jsonObject([
                "error": "file_too_large",
                "path": rawPath,
                "max_bytes": Self.maxReadBytes
            ])
        }
        guard let data = try? Data(contentsOf: file) else {
            return jsonObject(["error": "file_unreadable", "path": rawPath])
        }
        if data.prefix(8192).contains(0) {
            return jsonObject(["error": "binary_file", "path": rawPath])
        }
        guard let content = String(data: data, encoding: .utf8) else {
            return jsonObject(["error": "not_utf8", "path": rawPath])
        }
        let requested = integer(arguments, key: "max_chars", default: Self.maxReadCharacters)
        let limit = min(max(requested, 1), Self.maxReadCharacters)
        let truncated = content.count > limit
        return jsonObject([
            "path": rawPath,
            "content": String(content.prefix(limit)),
            "truncated": truncated,
            "bytes": data.count
        ])
    }

    private func searchFiles(_ arguments: [String: Any], root: URL) -> String {
        let query = string(arguments, key: "query")
        guard !query.isEmpty else { return jsonObject(["error": "empty_query"]) }
        let rawPath = string(arguments, key: "path")
        guard let searchRoot = resolvedPath(rawPath, root: root) else {
            return jsonObject(["error": "path_outside_workspace", "path": rawPath])
        }
        let caseSensitive = arguments["case_sensitive"] as? Bool ?? false
        let resultLimit = min(max(integer(arguments, key: "max_results", default: 50), 1), Self.maxResults)
        let needle = caseSensitive ? query : query.lowercased()
        var matches: [[String: Any]] = []
        guard let enumerator = FileManager.default.enumerator(
            at: searchRoot,
            includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .isSymbolicLinkKey],
            options: [.skipsPackageDescendants]
        ) else {
            return jsonObject(["error": "directory_unreadable", "path": rawPath])
        }
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            if Self.excludedDirectoryNames.contains(name) {
                enumerator.skipDescendants()
                continue
            }
            if Self.excludedFileNames.contains(name) { continue }
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey, .isSymbolicLinkKey])
            if values?.isDirectory == true || values?.isSymbolicLink == true { continue }
            guard let size = values?.fileSize, size <= Self.maxSearchBytes,
                  let data = try? Data(contentsOf: url),
                  !data.prefix(8192).contains(0),
                  let content = String(data: data, encoding: .utf8)
            else { continue }
            for (index, line) in content.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let text = String(line)
                let haystack = caseSensitive ? text : text.lowercased()
                guard haystack.contains(needle) else { continue }
                let relative = relativePath(url, from: root)
                matches.append([
                    "path": relative,
                    "line": index + 1,
                    "snippet": String(text.prefix(500))
                ])
                if matches.count >= resultLimit { break }
            }
            if matches.count >= resultLimit { break }
        }
        return jsonObject([
            "query": query,
            "matches": matches,
            "truncated": matches.count >= resultLimit
        ])
    }

    private func writeFile(_ arguments: [String: Any], root: URL) -> String {
        let rawPath = string(arguments, key: "path")
        let content = string(arguments, key: "content")
        guard let file = resolvedPath(rawPath, root: root) else {
            return jsonObject(["error": "path_outside_workspace", "path": rawPath])
        }
        guard !Self.excludedFileNames.contains(file.lastPathComponent) else {
            return jsonObject(["error": "sensitive_file_blocked", "path": rawPath])
        }
        guard content.count <= Self.maxWriteCharacters else {
            return jsonObject(["error": "content_too_large", "max_chars": Self.maxWriteCharacters])
        }
        do {
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data(content.utf8).write(to: file, options: .atomic)
            return jsonObject(["path": rawPath, "written": true, "bytes": Data(content.utf8).count])
        } catch {
            return jsonObject(["error": "write_failed", "detail": error.localizedDescription])
        }
    }

    private func inspectRepo(root: URL) async -> String {
        let markers = [
            "Package.swift", "pyproject.toml", "setup.py", "package.json",
            "Cargo.toml", "go.mod", "Makefile", ".gitignore"
        ]
        let found = markers.filter {
            FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path)
        }
        let result = await commandResult(
            "git branch --show-current",
            root: root,
            timeout: 10
        )
        let topLevel = (try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ))?
            .map(\.lastPathComponent)
            .filter { !Self.excludedDirectoryNames.contains($0) }
            .sorted()
            .prefix(100)
            .map { $0 } ?? []
        return jsonObject([
            "root": root.path,
            "is_git_repository": FileManager.default.fileExists(
                atPath: root.appendingPathComponent(".git").path
            ),
            "branch": result.exitCode == 0 ? result.stdout.trimmingCharacters(in: .whitespacesAndNewlines) : "",
            "project_markers": found,
            "top_level": topLevel,
            "git_error": result.exitCode == 0 ? "" : result.stderr
        ])
    }

    private func gitStatus(root: URL) async -> String {
        let result = await commandResult("git status --short --branch", root: root, timeout: 10)
        return jsonObject([
            "command": "git status --short --branch",
            "exit_code": result.exitCode,
            "stdout": result.stdout,
            "stderr": result.stderr
        ])
    }

    private func gitDiff(_ arguments: [String: Any], root: URL) async -> String {
        let scope = string(arguments, key: "scope").lowercased()
        let rawPath = string(arguments, key: "path")
        guard ["", "unstaged", "staged", "both"].contains(scope) else {
            return jsonObject([
                "error": "invalid_scope",
                "note": "git_diff scope must be unstaged, staged, or both."
            ])
        }
        guard rawPath.isEmpty || resolvedPath(rawPath, root: root) != nil else {
            return jsonObject(["error": "path_outside_workspace", "path": rawPath])
        }
        let pathSuffix = rawPath.isEmpty ? "" : " -- " + Self.shellQuote(rawPath)
        let commands: [(String, String)] = {
            switch scope {
            case "staged": return [("staged", "git diff --cached --no-ext-diff --unified=3\(pathSuffix)")]
            case "both": return [
                ("unstaged", "git diff --no-ext-diff --unified=3\(pathSuffix)"),
                ("staged", "git diff --cached --no-ext-diff --unified=3\(pathSuffix)")
            ]
            default: return [("unstaged", "git diff --no-ext-diff --unified=3\(pathSuffix)")]
            }
        }()
        var sections: [[String: Any]] = []
        for (name, command) in commands {
            let result = await commandResult(command, root: root, timeout: 20)
            sections.append([
                "scope": name,
                "command": command,
                "exit_code": result.exitCode,
                "diff": result.stdout,
                "stderr": result.stderr
            ])
        }
        return jsonObject([
            "path": rawPath,
            "scopes": sections
        ])
    }

    private func applyPatch(_ arguments: [String: Any], root: URL) async -> String {
        let patch = string(arguments, key: "patch")
        guard !patch.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return jsonObject(["error": "empty_patch"])
        }
        guard patch.count <= 2_000_000 else {
            return jsonObject(["error": "patch_too_large", "max_chars": 2_000_000])
        }
        let result = await Task.detached(priority: .userInitiated) {
            Self.applyPatchSynchronously(patch, root: root)
        }.value
        return jsonObject([
            "validated": result.validated,
            "applied": result.applied,
            "exit_code": result.exitCode,
            "stdout": result.stdout,
            "stderr": result.stderr
        ])
    }

    private func runTests(_ arguments: [String: Any], root: URL) async -> String {
        let requested = string(arguments, key: "command").trimmingCharacters(in: .whitespacesAndNewlines)
        let command = requested.isEmpty ? detectedTestCommand(root: root) : requested
        guard let command else {
            return jsonObject([
                "error": "no_test_runner_detected",
                "note": "Pass an explicit test command."
            ])
        }
        let result = await commandResult(command, root: root, timeout: 300)
        return jsonObject([
            "command": command,
            "exit_code": result.exitCode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timedOut
        ])
    }

    private func detectedTestCommand(root: URL) -> String? {
        let fm = FileManager.default
        if fm.fileExists(atPath: root.appendingPathComponent("pyproject.toml").path)
            || fm.fileExists(atPath: root.appendingPathComponent("pytest.ini").path)
            || fm.fileExists(atPath: root.appendingPathComponent("tests").path) {
            return "pytest -q"
        }
        if fm.fileExists(atPath: root.appendingPathComponent("Package.swift").path) {
            return "swift test"
        }
        if fm.fileExists(atPath: root.appendingPathComponent("package.json").path) {
            return "npm test"
        }
        return nil
    }

    private func commandResult(
        _ command: String,
        root: URL,
        timeout: TimeInterval
    ) async -> CommandResult {
        await Task.detached(priority: .userInitiated) {
            Self.executeCommand(command, root: root, timeout: timeout)
        }.value
    }

    private func runCommand(_ arguments: [String: Any], root: URL) async -> String {
        let command = string(arguments, key: "command").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !command.isEmpty else { return jsonObject(["error": "empty_command"]) }
        let timeout = min(max(integer(arguments, key: "timeout_seconds", default: 60), 1), 60)
        let result = await Task.detached(priority: .userInitiated) {
            Self.executeCommand(command, root: root, timeout: TimeInterval(timeout))
        }.value
        return jsonObject([
            "command": command,
            "exit_code": result.exitCode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timedOut
        ])
    }

    private struct CommandResult: Sendable {
        let exitCode: Int
        let stdout: String
        let stderr: String
        let timedOut: Bool
    }

    private static func executeCommand(_ command: String, root: URL, timeout: TimeInterval) -> CommandResult {
        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", command]
        process.currentDirectoryURL = root
        process.environment = [
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
            "HOME": NSHomeDirectory(),
            "LANG": "en_US.UTF-8",
            "PWD": root.path
        ]
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        do {
            try process.run()
        } catch {
            return CommandResult(exitCode: -1, stdout: "", stderr: error.localizedDescription, timedOut: false)
        }

        let stdoutCollector = OutputCollector(limit: 40_000)
        let stderrCollector = OutputCollector(limit: 20_000)
        let readers = DispatchGroup()
        readers.enter()
        DispatchQueue.global(qos: .utility).async {
            Self.collect(stdoutPipe.fileHandleForReading, into: stdoutCollector)
            readers.leave()
        }
        readers.enter()
        DispatchQueue.global(qos: .utility).async {
            Self.collect(stderrPipe.fileHandleForReading, into: stderrCollector)
            readers.leave()
        }
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        var timedOut = false
        if process.isRunning {
            timedOut = true
            process.terminate()
            Thread.sleep(forTimeInterval: 0.25)
            if process.isRunning {
                process.interrupt()
            }
        }
        process.waitUntilExit()
        readers.wait()
        return CommandResult(
            exitCode: Int(process.terminationStatus),
            stdout: stdoutCollector.string,
            stderr: stderrCollector.string,
            timedOut: timedOut
        )
    }

    private struct PatchResult: Sendable {
        let validated: Bool
        let applied: Bool
        let exitCode: Int
        let stdout: String
        let stderr: String
    }

    private static func applyPatchSynchronously(_ patch: String, root: URL) -> PatchResult {
        let patchURL = root.appendingPathComponent(".mtplx-patch-\(UUID().uuidString).diff")
        do {
            try Data(patch.utf8).write(to: patchURL, options: .atomic)
            let quotedPath = Self.shellQuote(patchURL.path)
            let check = executeCommand(
                "git apply --check --whitespace=nowarn -- \(quotedPath)",
                root: root,
                timeout: 60
            )
            guard check.exitCode == 0 else {
                try? FileManager.default.removeItem(at: patchURL)
                return PatchResult(
                    validated: false,
                    applied: false,
                    exitCode: check.exitCode,
                    stdout: check.stdout,
                    stderr: check.stderr
                )
            }
            let applied = executeCommand(
                "git apply --whitespace=nowarn -- \(quotedPath)",
                root: root,
                timeout: 60
            )
            try? FileManager.default.removeItem(at: patchURL)
            return PatchResult(
                validated: true,
                applied: applied.exitCode == 0,
                exitCode: applied.exitCode,
                stdout: applied.stdout,
                stderr: applied.stderr
            )
        } catch {
            try? FileManager.default.removeItem(at: patchURL)
            return PatchResult(
                validated: false,
                applied: false,
                exitCode: -1,
                stdout: "",
                stderr: error.localizedDescription
            )
        }
    }

    private static func collect(_ handle: FileHandle, into collector: OutputCollector) {
        while true {
            let data = handle.readData(ofLength: 8_192)
            if data.isEmpty { break }
            collector.append(data)
        }
    }

    private final class OutputCollector: @unchecked Sendable {
        private let limit: Int
        private let lock = NSLock()
        private var data = Data()

        init(limit: Int) {
            self.limit = limit
        }

        func append(_ value: Data) {
            lock.lock()
            defer { lock.unlock() }
            guard data.count < limit else { return }
            data.append(value.prefix(limit - data.count))
        }

        var string: String {
            lock.lock()
            defer { lock.unlock() }
            return String(data: data, encoding: .utf8) ?? ""
        }
    }

    private func approvalRequired(name: String) -> String {
        jsonObject([
            "error": "approval_required",
            "tool": name,
            "note": "The user must approve this workspace action before it can run."
        ])
    }

    private static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private func relativePath(_ url: URL, from root: URL) -> String {
        let canonicalRoot = root.path
        let alternateRoot: String
        if canonicalRoot.hasPrefix("/private/") {
            alternateRoot = String(canonicalRoot.dropFirst("/private".count))
        } else {
            alternateRoot = "/private" + canonicalRoot
        }
        for rootPath in [canonicalRoot, alternateRoot] {
            let prefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"
            if url.path.hasPrefix(prefix) {
                return String(url.path.dropFirst(prefix.count))
            }
        }
        return url.lastPathComponent
    }

    private func definition(
        name: String,
        description: String,
        properties: [String: JSONValue],
        required: [String]
    ) -> ChatRequestTool {
        ChatRequestTool(
            function: ChatRequestToolDefinition(
                name: name,
                description: description,
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object(properties),
                    "required": .array(required.map(JSONValue.string))
                ])
            )
        )
    }

    private func stringProperty(_ description: String) -> JSONValue {
        .object(["type": .string("string"), "description": .string(description)])
    }

    private func integerProperty(_ description: String) -> JSONValue {
        .object(["type": .string("integer"), "description": .string(description)])
    }

    private func booleanProperty(_ description: String) -> JSONValue {
        .object(["type": .string("boolean"), "description": .string(description)])
    }

    private func parseObject(_ json: String) -> [String: Any] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return object
    }

    private func string(_ object: [String: Any], key: String) -> String {
        object[key] as? String ?? ""
    }

    private func integer(_ object: [String: Any], key: String, default fallback: Int) -> Int {
        if let value = object[key] as? Int { return value }
        if let value = object[key] as? NSNumber { return value.intValue }
        return fallback
    }

    private func jsonObject(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8)
        else { return "{\"error\":\"json_encode_failed\"}" }
        return text
    }
}
