import SwiftUI
import MTPLXAppCore

// MARK: - LaunchOverlay
//
// Custom drop-down attached to the Play button. Lives in the
// `ContentView` ZStack as a top-right anchored overlay so it floats
// above the rest of the UI without disturbing layout. Dismisses on
// outside-tap (a transparent backdrop layer captures clicks).
//
// Animation choreography (driven by `presented`):
//   0   ms │ surface scales from 0.92 + opacity 0; border-trim at 0
//  50   ms │ border-stroke trims 0→1 (rectangle stroke draws clockwise)
// 220   ms │ header fades + slides down 6pt
// 300+60*i │ row i flicks in (opacity + 8pt drop) + its top divider
//          │ wipes left→right
//
// Picking `.other` collapses the row list to a custom-client config
// form (port + API key + endpoint preview + Start button).

struct LaunchOverlay: View, Equatable {
    let backend: MTPLXBackendStore
    let configuration: MTPLXAppConfiguration

    @EnvironmentObject private var themeStore: ThemeStore
    @EnvironmentObject private var router: AppRouter

    @Binding var presented: Bool
    private let presentedValue: Bool

    init(
        backend: MTPLXBackendStore,
        configuration: MTPLXAppConfiguration,
        presented: Binding<Bool>
    ) {
        self.backend = backend
        self.configuration = configuration
        _presented = presented
        presentedValue = presented.wrappedValue
    }

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.presentedValue == rhs.presentedValue
            && lhs.configuration == rhs.configuration
    }

    @State private var borderProgress: CGFloat = 0
    @State private var headerVisible: Bool = false
    @State private var rowsVisibleCount: Int = 0
    @State private var otherExpanded: Bool = false
    @State private var customPort: Int = 8000
    @State private var customApiKey: String = ""

    private let popoverWidth: CGFloat = 296
    private let cornerRadius: CGFloat = 12
    // Strip is 48pt tall; LaunchButton is a 32pt circle, 14pt from
    // right edge — its centre lands at right-30. Notch sits 8pt from
    // the popover's right edge, so rightOffset = 30 - 8 = 22 lines
    // the notch directly under the button.
    private let topOffset: CGFloat = 50
    private let rightOffset: CGFloat = 22

    var body: some View {
        ZStack(alignment: .topTrailing) {
            backdrop
            if presented {
                popoverColumn
                    .padding(.top, topOffset)
                    .padding(.trailing, rightOffset)
                    .transition(.identity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        .allowsHitTesting(presented)
        .onChange(of: presented) { _, isOn in
            if isOn {
                customPort = configuration.port
                customApiKey = configuration.apiKey ?? ""
                otherExpanded = false
                runEnterChoreography()
            } else {
                runExitChoreography()
            }
        }
    }

    // MARK: - Layers

    @ViewBuilder
    private var backdrop: some View {
        Color.clear
            .contentShape(Rectangle())
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .onTapGesture { presented = false }
            .opacity(presented ? 1 : 0)
    }

    @ViewBuilder
    private var popoverColumn: some View {
        VStack(alignment: .trailing, spacing: 0) {
            notch
            popoverSurface
                .frame(width: popoverWidth)
        }
        .frame(width: popoverWidth, alignment: .trailing)
        .opacity(borderProgress)
        .scaleEffect(borderProgress > 0 ? 1 : 0.94, anchor: .topTrailing)
    }

    @ViewBuilder
    private var notch: some View {
        DownNotch()
            .fill(Brand.raisedSurface)
            .frame(width: 12, height: 7)
            .padding(.trailing, 8)
            .padding(.bottom, -1)
            .opacity(borderProgress > 0.3 ? 1 : 0)
    }

    @ViewBuilder
    private var popoverSurface: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if otherExpanded {
                customClientForm
            } else {
                ForEach(Array(LaunchTarget.allCases.enumerated()), id: \.element.id) { idx, target in
                    if idx > 0 {
                        rowDivider(precedesRow: idx)
                    }
                    LaunchRow(
                        target: target,
                        index: idx,
                        visible: rowsVisibleCount > idx,
                        isLast: configuration.lastLaunchTarget == target.rawValue,
                        motionEnabled: motionEnabled,
                        onPick: handlePick
                    )
                }
            }
        }
        .padding(.vertical, 6)
        .animation(motionEnabled ? .smooth(duration: 0.30) : nil, value: otherExpanded)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Brand.raisedSurface)
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .trim(from: 0, to: borderProgress)
                    .stroke(Brand.separatorStrong, lineWidth: 0.75)
            }
            .shadow(color: .black.opacity(0.55), radius: 18, x: 0, y: 10)
        )
    }

    // MARK: - Header

    @ViewBuilder
    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(otherExpanded ? "Custom client" : "Start MTPLX")
                    .font(.system(.callout, design: .rounded).weight(.semibold))
                    .foregroundStyle(Brand.typeBody)
                Text(otherExpanded
                     ? "Point any OpenAI- or Anthropic-compatible app at MTPLX."
                     : "Pick how you want to use it.")
                    .font(.caption2)
                    .foregroundStyle(Brand.typeTertiary)
            }
            Spacer(minLength: 0)
            if otherExpanded {
                Button { otherExpanded = false } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Brand.typeSecondary)
                        .frame(width: 22, height: 18)
                        .background(
                            Capsule(style: .continuous)
                                .stroke(Brand.separator, lineWidth: 0.75)
                        )
                }
                .buttonStyle(.plain)
                .help("Back")
            }
        }
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .opacity(headerVisible ? 1 : 0)
        .offset(y: headerVisible ? 0 : -6)
    }

    // MARK: - Divider

    @ViewBuilder
    private func rowDivider(precedesRow row: Int) -> some View {
        let visible = rowsVisibleCount > row
        Rectangle()
            .fill(Brand.separator)
            .frame(height: 0.5)
            .scaleEffect(x: visible ? 1 : 0, y: 1, anchor: .leading)
            .opacity(visible ? 1 : 0)
    }

    // MARK: - Custom-client form

    @ViewBuilder
    private var customClientForm: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Port
            HStack(spacing: 10) {
                Text("Port")
                    .font(.system(.callout).weight(.medium))
                    .foregroundStyle(Brand.typeBody)
                    .frame(width: 64, alignment: .leading)
                TextField("8000",
                          value: $customPort,
                          format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.callout, design: .monospaced))
                    .frame(maxWidth: 100)
                Spacer()
            }

            // API key
            HStack(spacing: 10) {
                Text("API key")
                    .font(.system(.callout).weight(.medium))
                    .foregroundStyle(Brand.typeBody)
                    .frame(width: 64, alignment: .leading)
                SecureField("optional", text: $customApiKey)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(.callout, design: .monospaced))
            }

            Divider().overlay(Brand.separator)

            // Connection details
            VStack(alignment: .leading, spacing: 8) {
                Text("CONNECT YOUR CLIENT TO")
                    .font(.system(size: 9, weight: .heavy, design: .monospaced))
                    .tracking(1.5)
                    .foregroundStyle(Brand.typeTertiary)
                endpointRow(label: "OpenAI", url: openAIBase)
                endpointRow(label: "Anthropic", url: anthropicBase)
            }

            Button {
                startCustom()
            } label: {
                HStack {
                    Spacer()
                    Text("Start serving")
                        .font(.system(.callout, design: .rounded).weight(.semibold))
                    Image(systemName: "play.fill")
                        .font(.system(size: 10, weight: .semibold))
                    Spacer()
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 14)
                .background(
                    Capsule(style: .continuous)
                        .fill(Brand.success.opacity(0.18))
                        .overlay(
                            Capsule(style: .continuous)
                                .stroke(Brand.success.opacity(0.55), lineWidth: 0.75)
                        )
                )
                .foregroundStyle(Brand.success)
            }
            .buttonStyle(.plain)
            .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
    }

    @ViewBuilder
    private func endpointRow(label: String, url: String) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(Brand.typeSecondary)
                .frame(width: 64, alignment: .leading)
            Text(url)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(Brand.typeBody)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
            Button {
                copyToPasteboard(url)
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(Brand.typeSecondary)
            }
            .buttonStyle(.plain)
            .help("Copy")
        }
    }

    private var openAIBase: String { "http://127.0.0.1:\(customPort)/v1" }
    private var anthropicBase: String { "http://127.0.0.1:\(customPort)" }

    private func copyToPasteboard(_ text: String) {
        #if canImport(AppKit)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #endif
    }

    private func startCustom() {
        var config = backend.configuration
        config.port = customPort
        config.apiKey = customApiKey.isEmpty ? nil : customApiKey
        try? backend.saveSettings(config)
        presented = false
        Task { await backend.startDaemon(target: .other) }
    }

    // MARK: - Choreography

    private var motionEnabled: Bool {
        !configuration.performanceLock && !themeStore.reduceMotionPreference
    }

    private func handlePick(_ target: LaunchTarget) {
        if target == .other {
            // Expand the inline form rather than spawning the daemon.
            // The form has its own Start button which then dispatches.
            if motionEnabled {
                withAnimation(.smooth(duration: 0.28)) {
                    otherExpanded = true
                }
            } else {
                otherExpanded = true
            }
            return
        }
        if target == .benchmark {
            // Benchmark opens the native overlay immediately. Its Run
            // button owns daemon readiness so the user sees startup,
            // model-download prompts, and AIME progress in one place.
            presented = false
            router.openBenchmark()
            return
        }
        if target == .hermes {
            presented = false
            Task { await backend.startDaemon(target: .hermes) }
            return
        }
        if target == .chat {
            presented = false
            if motionEnabled {
                withAnimation(.smooth(duration: 0.24)) {
                    router.showChat()
                }
            } else {
                router.showChat()
            }
            Task { await backend.startDaemon(target: target) }
            return
        }
        presented = false
        Task { await backend.startDaemon(target: target) }
    }

    private func runEnterChoreography() {
        OverlayChoreography.runEnter(
            motionEnabled: motionEnabled,
            rowCount: LaunchTarget.allCases.count,
            borderProgress: $borderProgress,
            headerVisible: $headerVisible,
            rowsVisibleCount: $rowsVisibleCount
        )
    }

    private func runExitChoreography() {
        OverlayChoreography.runExit(
            motionEnabled: motionEnabled,
            borderProgress: $borderProgress,
            headerVisible: $headerVisible,
            rowsVisibleCount: $rowsVisibleCount,
            additionalReset: { otherExpanded = false }
        )
    }
}

// MARK: - LaunchRow
//
// Per-row view. Owns its own hover state so each row can magnify +
// pop with its own 3D shadow without re-rendering its siblings.

private struct LaunchRow: View {
    let target: LaunchTarget
    let index: Int
    let visible: Bool
    let isLast: Bool
    let motionEnabled: Bool
    let onPick: (LaunchTarget) -> Void

    var body: some View {
        Button {
            onPick(target)
        } label: {
            HStack(spacing: 10) {
                TargetIcon(target: target, isLast: isLast)
                    .frame(width: 22, height: 18)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(target.title)
                            .font(.system(.callout, design: .rounded).weight(.medium))
                            .foregroundStyle(Brand.typeBody)
                        if isLast {
                            Text("LAST")
                                .font(.system(size: 8, weight: .heavy, design: .monospaced))
                                .tracking(1)
                                .foregroundStyle(Brand.typeBody)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(
                                    Capsule(style: .continuous)
                                        .stroke(Brand.separatorStrong, lineWidth: 0.75)
                                )
                        }
                    }
                    Text(target.tagline)
                        .font(.caption2)
                        .foregroundStyle(Brand.typeTertiary)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                if target.hasInlineForm {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(Brand.typeTertiary)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .overlayRowHover(motionEnabled: motionEnabled)
        .opacity(visible ? 1 : 0)
        .offset(y: visible ? 0 : 8)
    }
}

// MARK: - DownNotch shape

private struct DownNotch: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

// MARK: - TargetIcon
//
// Renders the launch-target icon. Every target uses its SF Symbol
// except `.openCode` — that one renders the actual OpenCode app mark
// (a thick rounded-rect "square donut" with a hole) via the custom
// `SquareDonut` shape filled with even-odd rule. SF Symbols don't
// ship a true square-donut variant; `square` is too thin, `app.fill`
// is too solid.

private struct TargetIcon: View {
    let target: LaunchTarget
    let isLast: Bool

    var body: some View {
        if target == .openCode {
            OpenCodeMark(
                frameColor: isLast ? Brand.accent : Brand.typeSecondary,
                detailColor: (isLast ? Brand.accent : Brand.typeSecondary).opacity(0.45)
            )
            .frame(width: 12, height: 15)
        } else if target == .hermes {
            // Real Hermes Agent (Nous Research) brand mark instead of a
            // generic glyph. Tinted to match the other launch-row icons —
            // neutral gray at rest, accent when it's the last-used target —
            // so it doesn't read as a too-bright white blob in the list.
            HermesMark(size: 16, tint: isLast ? Brand.accent : Brand.typeSecondary)
        } else {
            Image(systemName: target.systemImage)
                .font(.system(size: 13, weight: .regular))
                .symbolRenderingMode(.monochrome)
                .foregroundStyle(isLast ? Brand.accent : Brand.typeSecondary)
        }
    }
}

// MARK: - OpenCodeMark
//
// 1:1 SwiftUI reconstruction of the official OpenCode mark from
// dashboardicons.com/icons/opencode (verified against opencode.ai/brand).
//
// SVG viewBox = 240 × 300 (W:H = 4:5). Geometry:
//   - Outer canvas      : 240 × 300, sharp corners
//   - Inner cutout      : 120 × 180 centred at (60,60) → (180,240)
//                         — uniform 60pt border (25% of width / 20% of
//                         height)
//   - Detail fill       : 120 × 120 in the bottom 2/3 of the cutout,
//                         (60,120) → (180,240), contrast color
//
// The mark is sharp-cornered; the rounded look in some renders is the
// macOS icon wrapper, not the mark.
//
// Rendered with `Canvas` so the frame uses even-odd fill (canvas
// minus inner) in `frameColor`, and the detail rect uses
// `detailColor` painted underneath.

private struct OpenCodeMark: View {
    let frameColor: Color
    let detailColor: Color

    var body: some View {
        Canvas { context, size in
            let sx = size.width / 240
            let sy = size.height / 300

            // Detail fill — bottom 2/3 of the inner cutout.
            let detail = CGRect(
                x: 60 * sx,
                y: 120 * sy,
                width: 120 * sx,
                height: 120 * sy
            )
            context.fill(Path(detail), with: .color(detailColor))

            // Frame — full canvas minus the 120×180 inner cutout,
            // even-odd filled so the cutout punches through.
            var framePath = Path()
            framePath.addRect(CGRect(origin: .zero, size: size))
            framePath.addRect(CGRect(
                x: 60 * sx,
                y: 60 * sy,
                width: 120 * sx,
                height: 180 * sy
            ))
            context.fill(framePath, with: .color(frameColor), style: FillStyle(eoFill: true))
        }
    }
}
