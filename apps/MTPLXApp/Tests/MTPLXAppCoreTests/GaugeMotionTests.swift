import SwiftUI
import XCTest
@testable import MTPLXAppCore

/// The speedometer's motion, solved on its own timeline, must move as the
/// SwiftUI springs it replaces did: same springs, same overshoot, no jump
/// when a new reading lands mid-move, and at rest once nothing visible is
/// left of a move.
final class GaugeMotionTests: XCTestCase {
    private let tolerance = AdditiveSpringValue.restTolerance

    // MARK: - The springs are SwiftUI's

    /// The needle ran `.interpolatingSpring(mass: 0.6, stiffness: 140,
    /// damping: 22)`: SwiftUI's spring with those raw parameters, damping
    /// ratio 1.2, overdamping allowed.
    func testNeedleSpringMatchesSwiftUIsSpring() {
        let ours = DampedSpring(mass: 0.6, stiffness: 140, damping: 22)
        let theirs = Spring(mass: 0.6, stiffness: 140, damping: 22, allowOverDamping: true)
        XCTAssertEqual(theirs.dampingRatio, 1.2, accuracy: 0.001)
        for step in 0...480 {
            let t = Double(step) / 240
            let displacement = 1 - theirs.value(target: 1.0, time: t)
            XCTAssertEqual(ours.state(x0: 1, v0: 0, at: t).x, displacement, accuracy: 1e-9, "t=\(t)")
        }
    }

    /// The flick and the digit roll ran `.spring(response: 0.28,
    /// dampingFraction: 0.72)`, underdamped: it overshoots.
    func testFlickSpringMatchesSwiftUIsSpringWithAndWithoutVelocity() {
        let ours = DampedSpring(response: 0.28, dampingFraction: 0.72)
        let theirs = Spring(response: 0.28, dampingRatio: 0.72)
        for velocity in [0.0, 3.0, -2.0] {
            for step in 0...240 {
                let t = Double(step) / 240
                let displacement = 1 - theirs.value(target: 1.0, initialVelocity: velocity, time: t)
                XCTAssertEqual(
                    ours.state(x0: 1, v0: -velocity, at: t).x, displacement,
                    accuracy: 1e-9, "t=\(t) v=\(velocity)"
                )
            }
        }
        let overshoot = (0...240).map { ours.state(x0: 1, v0: 0, at: Double($0) / 240).x }.min() ?? 0
        XCTAssertLessThan(overshoot, -0.03, "the roll and the flick overshoot, as they did")
    }

    func testVelocityIsTheRateOfChangeOfPosition() {
        let springs = [
            DampedSpring(mass: 0.6, stiffness: 140, damping: 22),
            DampedSpring(response: 0.28, dampingFraction: 0.72),
            DampedSpring(mass: 1, stiffness: 100, damping: 20),
        ]
        let h = 1e-6
        for spring in springs {
            for t in [0.0, 0.05, 0.2, 0.5] {
                let ahead = spring.state(x0: 0.7, v0: -1.5, at: t + h).x
                let behind = spring.state(x0: 0.7, v0: -1.5, at: max(0, t - h)).x
                let rate = (ahead - behind) / (t == 0 ? h : 2 * h)
                XCTAssertEqual(spring.state(x0: 0.7, v0: -1.5, at: t).v, rate, accuracy: 1e-3, "t=\(t)")
            }
        }
    }

    func testRestTimeIsWhenNothingVisibleIsLeft() {
        let spring = DampedSpring(response: 0.28, dampingFraction: 0.72)
        let rest = spring.restTime(x0: 35, v0: 0, tolerance: tolerance)
        XCTAssertGreaterThan(rest, 0.2)
        XCTAssertLessThan(rest, 0.5)
        for step in 0...960 {
            let t = rest + Double(step) / 240
            let state = spring.state(x0: 35, v0: 0, at: t)
            XCTAssertLessThan(abs(state.x), tolerance, "t=\(t)")
            XCTAssertLessThan(abs(state.v) / 60, tolerance, "t=\(t)")
        }
    }

    // MARK: - The needle

    func testANewReadingMidMoveNeverJerksTheNeedle() throws {
        var arc = AdditiveSpringValue(spring: DampedSpring(mass: 0.6, stiffness: 140, damping: 22), pointsPerUnit: 600)
        arc.move(to: 0.2, at: 0, animated: false)
        arc.move(to: 0.5, at: 1, animated: true)
        let before = arc.value(at: 1.2)
        let slope = (arc.value(at: 1.2 + 1e-6) - arc.value(at: 1.2 - 1e-6)) / 2e-6
        arc.move(to: 0.4, at: 1.2, animated: true)
        XCTAssertEqual(arc.value(at: 1.2), before, accuracy: 1e-12)
        let slopeAfter = (arc.value(at: 1.2 + 1e-6) - arc.value(at: 1.2)) / 1e-6
        XCTAssertEqual(slopeAfter, slope, accuracy: 1e-2, "the needle keeps its speed through a new reading")

        let rest = try XCTUnwrap(arc.restTime)
        XCTAssertEqual(arc.value(at: rest), 0.4, accuracy: tolerance / 600)
        XCTAssertEqual(arc.value(at: rest + 0.01), 0.4)
    }

    func testAMoveTooSmallToSeeLandsAtOnce() {
        var arc = AdditiveSpringValue(spring: DampedSpring(mass: 0.6, stiffness: 140, damping: 22), pointsPerUnit: 600)
        arc.move(to: 0.5, at: 0, animated: false)
        arc.move(to: 0.5 + 0.1 / 600, at: 1, animated: true)
        XCTAssertNil(arc.restTime)
        XCTAssertEqual(arc.value(at: 1), 0.5 + 0.1 / 600)
    }

    func testATypicalDecodeStepRestsWellInsideTheHalfSecondBetweenReadings() throws {
        // 82.4 to 83.1 tok/s on the 280 pt hero: the needle moves a few points.
        var arc = AdditiveSpringValue(spring: DampedSpring(mass: 0.6, stiffness: 140, damping: 22), pointsPerUnit: 600)
        arc.move(to: 0.70, at: 0, animated: false)
        arc.move(to: 0.705, at: 0, animated: true)
        let rest = try XCTUnwrap(arc.restTime)
        XCTAssertLessThan(rest, 0.45)
    }

    // MARK: - The flick

    func testFlickRisesToSixPercentAndComesBack() {
        var pulse = GaugePulse(pointsPerUnit: 100)
        pulse.fire(at: 10)
        let samples = (0...240).map { pulse.scale(at: 10 + Double($0) / 240) }
        let peak = samples.max() ?? 0
        XCTAssertEqual(peak, 1.06, accuracy: 0.005)
        XCTAssertLessThan(pulse.duration, 0.6)
        XCTAssertEqual(pulse.scale(at: pulse.end), 1)
        XCTAssertEqual(pulse.scale(at: pulse.end - 1e-4), 1, accuracy: tolerance / 100)
        XCTAssertEqual(pulse.scale(at: 9.9), 1, "no flick before it fires")
    }

    func testAFlickThatFiresMidFlightStartsWhereTheLastOneIs() {
        var pulse = GaugePulse(pointsPerUnit: 100)
        pulse.fire(at: 0)
        for t in [0.1, 0.35] {
            let current = pulse.state(at: t)
            var refired = pulse
            refired.fire(at: t)
            XCTAssertEqual(refired.state(at: t).scale, current.scale, accuracy: 1e-12)
            XCTAssertEqual(refired.state(at: t).velocity, current.velocity, accuracy: 1e-9)
        }
    }

    func testStoppedFlickIsAtRest() {
        var pulse = GaugePulse(pointsPerUnit: 100)
        pulse.fire(at: 0)
        pulse.stop()
        XCTAssertEqual(pulse.scale(at: 0.1), 1)
        XCTAssertEqual(pulse.end, 0)
    }

    // MARK: - The digit roll

    func testOnlyTheDigitsThatChangedRoll() {
        var roll = GaugeRoll(text: "82.4", travel: 35)
        roll.roll(to: "82.6", at: 10, animated: true)
        XCTAssertEqual((0..<4).map { roll.changes(at: $0) }, [false, false, false, true])
        XCTAssertEqual(roll.progress(at: 10), 0)
        XCTAssertTrue(roll.isRolling(at: 10.1))
        XCTAssertFalse(roll.isRolling(at: roll.end))
        XCTAssertEqual(roll.progress(at: roll.end), 1)
        XCTAssertLessThan(roll.end - 10, 0.5)
        // What is left of the travel at the end is under half a pixel.
        XCTAssertLessThan(abs(1 - roll.progress(at: roll.end - 1e-4)) * 35, tolerance)
    }

    /// A new digit lands first and sharpens after, as the counter's digits
    /// always did: its blur, fade and size run on a slower curve that never
    /// overshoots.
    func testANewDigitLandsThenSharpens() {
        var roll = GaugeRoll(text: "70.0", travel: 47)
        roll.roll(to: "40.0", at: 0, animated: true)
        XCTAssertEqual(roll.fade(at: 0), 0, accuracy: 1e-12)
        let landed = stride(from: 0.0, through: 0.5, by: 1.0 / 240)
            .first { abs(1 - roll.progress(at: $0)) * 47 < 1 } ?? 1
        XCTAssertLessThan(roll.fade(at: landed), 0.9, "still soft when it lands")
        var last = 0.0
        for step in 0...240 {
            let fade = roll.fade(at: Double(step) / 240)
            XCTAssertGreaterThanOrEqual(fade, last - 1e-12)
            XCTAssertLessThanOrEqual(fade, 1)
            last = fade
        }
        XCTAssertEqual(roll.fade(at: roll.end), 1)
        XCTAssertLessThan(roll.end, 0.45)
    }

    func testAChangeOfLengthRollsEveryDigit() {
        var roll = GaugeRoll(text: "99.9", travel: 35)
        roll.roll(to: "100", at: 0, animated: true)
        XCTAssertTrue((0..<3).allSatisfy { roll.changes(at: $0) })
    }

    func testTheSameTextDoesNotRollAgain() {
        var roll = GaugeRoll(text: "82.4", travel: 35)
        roll.roll(to: "82.6", at: 0, animated: true)
        let end = roll.end
        roll.roll(to: "82.6", at: 0.3, animated: true)
        XCTAssertEqual(roll.end, end)
        XCTAssertEqual(roll.from, "82.4")
    }

    func testWithoutAnimationTheTextLandsAtOnce() {
        var roll = GaugeRoll(text: "82.4", travel: 35)
        roll.roll(to: "82.6", at: 0, animated: true)
        roll.roll(to: "90.1", at: 0.1, animated: false)
        XCTAssertFalse(roll.isRolling(at: 0.1))
        XCTAssertEqual(roll.from, "90.1")
        XCTAssertEqual(roll.to, "90.1")

        roll.roll(to: "91.0", at: 1, animated: true)
        roll.settle()
        XCTAssertFalse(roll.isRolling(at: 1.05))
        XCTAssertEqual(roll.from, "91.0")
    }
}
