import Combine
import SwiftUI
import MTPLXAppCore

// MARK: - MenuBarReadout
//
// What the menu bar shows: the engine's state, how much of the prompt is
// read while a request reads it, and the live decode speed while it
// decodes. The status item (MenuBarController) observes only this, never
// the backend store, which publishes several times a second during a
// decode: the reading changes at most once a second, only when what is
// shown changes, and not at all while nothing runs, so the item costs
// nothing on an idle Mac and one title write a second at most during a
// request.

@MainActor
final class MenuBarReadout: ObservableObject {
    struct Reading: Equatable {
        var state: DaemonStateKind
        /// Whole tokens per second of a decode in progress; nil otherwise.
        var decodeTokS: Int?
        /// Whole percent of the prompt read while a request reads it; nil otherwise.
        var prefillPercent: Int? = nil
        /// A user's request is in flight on a running engine: the item keeps its
        /// wide length from the first token of the prompt to the last of the answer.
        var requestActive: Bool = false
    }

    @Published private(set) var reading = Reading(state: .stopped, decodeTokS: nil)
    private var subscription: AnyCancellable?

    func attach(to backend: MTPLXBackendStore) {
        guard subscription == nil else { return }
        subscription = backend.$headlineDecode
            .combineLatest(backend.$daemonState, backend.$inFlight)
            .map { decode, state, inFlight in
                Self.reading(decode: decode, state: state, inFlight: inFlight)
            }
            .removeDuplicates()
            .throttle(for: .seconds(1), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] reading in
                guard let self, self.reading != reading else { return }
                self.reading = reading
            }
    }

    /// The number in the menu bar means "decoding now": a running engine,
    /// a live reading, and a user's request in flight with tokens out (the
    /// same gate as the gauges, so the engine's own warm-up reading never
    /// shows). A held reading, the last turn's speed, is not shown.
    static func reading(decode: HeadlineDecodeReading, state: DaemonState, decoding: Bool) -> Reading {
        let kind = state.kind
        guard kind == .running, decoding, case .live(let value) = decode, value.isFinite, value >= 0.5 else {
            return Reading(state: kind, decodeTokS: nil)
        }
        return Reading(state: kind, decodeTokS: Int(value.rounded()))
    }

    /// The whole reading from what is in flight: the speed while a request
    /// decodes; while one reads its prompt (no tokens out yet), the percent
    /// of its prompt read, capped at 99 so "100%" never sits beside a first
    /// token still to come.
    static func reading(decode: HeadlineDecodeReading, state: DaemonState, inFlight: [InFlightRequest]) -> Reading {
        let kind = state.kind
        let decoding = inFlight.contains { $0.hasDecodeProgress }
        var reading = reading(decode: decode, state: state, decoding: decoding)
        guard kind == .running, !inFlight.isEmpty else { return reading }
        reading.requestActive = true
        if !decoding,
           let prefill = inFlight.lazy.compactMap(\.prefillState)
               .first(where: { $0.tokensTotal > 0 && $0.tokensDone != nil }) {
            reading.prefillPercent = min(99, max(0, Int((prefill.progress * 100).rounded(.down))))
        }
        return reading
    }
}
