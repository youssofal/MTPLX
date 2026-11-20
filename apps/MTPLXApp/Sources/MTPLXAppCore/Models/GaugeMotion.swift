import Foundation

// MARK: - GaugeMotion
//
// The speedometer's motion, solved in closed form instead of left to
// SwiftUI's animation engine. A SwiftUI spring runs its whole settling time
// whatever the distance, at the display's full rate: the arc's spring takes
// about 0.85 s to settle, the flick and the digit roll overlap it, and a
// decode delivers a reading every 0.5 s, so the window redrew at 120 Hz for
// the whole of every decode (39-53% of a core and 350-475 ms/s of
// WindowServer on the replay fixture, 12 November). The same springs solved
// here let the gauge draw on a timeline of its own: at 60 Hz, only while
// something moves, and not past the moment what is left of a move is under
// half a Retina pixel.

/// A damped spring: displacement and velocity at any time from any start.
/// The physics SwiftUI's spring animations run (unit-free: the caller picks
/// the units of `x0` and `v0`).
public struct DampedSpring: Equatable, Sendable {
    public let mass: Double
    public let stiffness: Double
    public let damping: Double

    public init(mass: Double = 1, stiffness: Double, damping: Double) {
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping
    }

    /// SwiftUI's `.spring(response:dampingFraction:)`: unit mass, a period of
    /// `response` seconds undamped.
    public init(response: Double, dampingFraction: Double) {
        let omega = 2 * Double.pi / response
        self.init(mass: 1, stiffness: omega * omega, damping: 2 * dampingFraction * omega)
    }

    /// Displacement from rest and velocity `t` seconds after starting at
    /// displacement `x0` with velocity `v0`.
    public func state(x0: Double, v0: Double, at t: Double) -> (x: Double, v: Double) {
        let t = max(0, t)
        let omega0 = (stiffness / mass).squareRoot()
        let zeta = damping / (2 * (stiffness * mass).squareRoot())
        if zeta < 1 - 1e-9 {
            let decay = zeta * omega0
            let omegaD = omega0 * (1 - zeta * zeta).squareRoot()
            let e = exp(-decay * t)
            let c = cos(omegaD * t)
            let s = sin(omegaD * t)
            let b = (v0 + decay * x0) / omegaD
            let x = e * (x0 * c + b * s)
            let v = e * (v0 * c - (decay * v0 + omega0 * omega0 * x0) / omegaD * s)
            return (x, v)
        }
        if zeta > 1 + 1e-9 {
            let root = (zeta * zeta - 1).squareRoot()
            let r1 = -omega0 * (zeta - root)
            let r2 = -omega0 * (zeta + root)
            let a = (v0 - r2 * x0) / (r1 - r2)
            let b = x0 - a
            let e1 = exp(r1 * t)
            let e2 = exp(r2 * t)
            return (a * e1 + b * e2, a * r1 * e1 + b * r2 * e2)
        }
        let e = exp(-omega0 * t)
        let k = v0 + omega0 * x0
        return (e * (x0 + k * t), e * (v0 - omega0 * k * t))
    }

    /// The time after which a move that starts at displacement `x0` with
    /// velocity `v0` stays within `tolerance` of rest and moves less than
    /// `tolerance` per 60 Hz frame. Same units as `x0`.
    public func restTime(x0: Double, v0: Double, tolerance: Double) -> Double {
        let step = 1.0 / 240.0
        let horizon = 4.0
        var lastMoving = -step
        var t = 0.0
        while t <= horizon {
            let (x, v) = state(x0: x0, v0: v0, at: t)
            if abs(x) >= tolerance || abs(v) / 60 >= tolerance {
                lastMoving = t
            }
            t += step
        }
        return min(horizon, lastMoving + step)
    }
}

// MARK: - The arc

/// The arc's value under SwiftUI's interpolating spring: a new reading adds
/// a move from the last target, and moves add up while they overlap, so a
/// new reading never jerks the arc. A move ends once what is left of it is
/// under `restTolerance` points.
public struct AdditiveSpringValue: Equatable, Sendable {
    public struct Move: Equatable, Sendable {
        public let start: Double
        public let delta: Double
        public let end: Double
    }

    /// Under half a pixel on a Retina display.
    public static let restTolerance = 0.25

    public let spring: DampedSpring
    /// Points on screen per unit of the value.
    public var pointsPerUnit: Double
    public private(set) var target: Double
    public private(set) var moves: [Move] = []

    public init(spring: DampedSpring, pointsPerUnit: Double, value: Double = 0) {
        self.spring = spring
        self.pointsPerUnit = pointsPerUnit
        self.target = value
    }

    /// Heads for `newTarget` from `time`; without animation the value lands
    /// there at once. A move too small to see lands at once too.
    public mutating func move(to newTarget: Double, at time: Double, animated: Bool) {
        moves.removeAll { $0.end <= time }
        let delta = newTarget - target
        target = newTarget
        guard animated else {
            moves.removeAll()
            return
        }
        let distance = abs(delta) * pointsPerUnit
        guard distance >= Self.restTolerance else { return }
        let duration = spring.restTime(x0: distance, v0: 0, tolerance: Self.restTolerance)
        moves.append(Move(start: time, delta: delta, end: time + duration))
    }

    public func value(at time: Double) -> Double {
        var value = target
        for move in moves where time < move.end {
            value -= move.delta * spring.state(x0: 1, v0: 0, at: time - move.start).x
        }
        return value
    }

    /// When the last move comes to rest; nil when nothing moves.
    public var restTime: Double? { moves.map(\.end).max() }
}

// MARK: - The flick

/// The hero number's flick on a new reading: up to 1.06 and back, the two
/// springs the phase animator ran (0.28 s response, 0.72 damping). The way
/// back starts when the way up is logically done, one response in, and
/// carries on from where the way up is and how fast it moves. A flick that
/// fires while one is still moving starts from where that one is, at its
/// speed, as a retargeted spring does.
public struct GaugePulse: Equatable, Sendable {
    public static let spring = DampedSpring(response: 0.28, dampingFraction: 0.72)
    public static let peak = 1.06
    public static let turnAround = 0.28

    /// Points on screen per unit of scale (half the number's width).
    public var pointsPerUnit: Double
    public private(set) var start: Double?
    public private(set) var end: Double = 0
    private var startScale = 1.0
    private var startVelocity = 0.0

    public init(pointsPerUnit: Double) {
        self.pointsPerUnit = pointsPerUnit
    }

    public mutating func fire(at time: Double) {
        let current = state(at: time)
        start = time
        startScale = current.scale
        startVelocity = current.velocity
        end = time + duration
    }

    public mutating func stop() {
        start = nil
        end = 0
        startScale = 1
        startVelocity = 0
    }

    public func scale(at time: Double) -> Double { state(at: time).scale }

    /// Scale and its rate of change at `time`.
    public func state(at time: Double) -> (scale: Double, velocity: Double) {
        guard let start, time < end else { return (1, 0) }
        let t = max(0, time - start)
        if t < Self.turnAround {
            let up = Self.spring.state(x0: startScale - Self.peak, v0: startVelocity, at: t)
            return (Self.peak + up.x, up.v)
        }
        let back = backPhase()
        let x = Self.spring.state(x0: back.x0, v0: back.v0, at: t - Self.turnAround)
        return (1 + x.x, x.v)
    }

    /// How long a flick moves for: the way up, then the way back until it
    /// is under `AdditiveSpringValue.restTolerance` points from rest.
    public var duration: Double {
        let back = backPhase()
        return Self.turnAround + Self.spring.restTime(
            x0: back.x0 * pointsPerUnit,
            v0: back.v0 * pointsPerUnit,
            tolerance: AdditiveSpringValue.restTolerance
        )
    }

    /// Where the way back starts: the way up's displacement from 1.0 and
    /// its velocity at the turnaround.
    private func backPhase() -> (x0: Double, v0: Double) {
        let up = Self.spring.state(x0: startScale - Self.peak, v0: startVelocity, at: Self.turnAround)
        return (Self.peak + up.x - 1, up.v)
    }
}

// MARK: - The digit roll

/// The digits' roll on a new reading: digits that changed slide up and out
/// while the new ones slide up and in, digits that did not change stay put.
/// Always upward, whichever way the reading went, as the counter has always
/// rolled. A changing digit moves on the flick's spring; its blur, fade and
/// size settle on a slower, critically damped one, so a new digit lands and
/// then sharpens, as it did.
public struct GaugeRoll: Equatable, Sendable {
    public static let spring = GaugePulse.spring
    public static let fadeSpring = DampedSpring(response: 0.3, dampingFraction: 1.0)
    /// The size a digit leaves at and arrives from.
    public static let minimumScale = 0.8

    public private(set) var from: String
    public private(set) var to: String
    public private(set) var start: Double?
    public private(set) var end: Double = 0
    /// Points the digits travel.
    public var travel: Double

    public init(text: String, travel: Double) {
        self.from = text
        self.to = text
        self.travel = travel
    }

    /// Rolls from what is shown now to `text`; without animation it lands at once.
    public mutating func roll(to text: String, at time: Double, animated: Bool) {
        guard text != to else { return }
        from = to
        to = text
        guard animated else {
            settle()
            return
        }
        start = time
        let tolerance = AdditiveSpringValue.restTolerance
        end = time + max(
            Self.spring.restTime(x0: travel, v0: 0, tolerance: tolerance),
            Self.fadeSpring.restTime(x0: travel * (1 - Self.minimumScale), v0: 0, tolerance: tolerance)
        )
    }

    /// Lands any roll in flight.
    public mutating func settle() {
        from = to
        start = nil
        end = 0
    }

    /// 0 when the roll starts, 1 at rest; overshoots a little on the way,
    /// as the spring does.
    public func progress(at time: Double) -> Double {
        guard let start, time < end else { return 1 }
        return 1 - Self.spring.state(x0: 1, v0: 0, at: time - start).x
    }

    /// 0 when the roll starts, 1 at rest, never past it: how far a changing
    /// digit has sharpened, faded in and grown to full size.
    public func fade(at time: Double) -> Double {
        guard let start, time < end else { return 1 }
        return min(1, max(0, 1 - Self.fadeSpring.state(x0: 1, v0: 0, at: time - start).x))
    }

    public func isRolling(at time: Double) -> Bool {
        start != nil && time < end
    }

    /// Whether character `index` changes in this roll: every one when the
    /// length changes (the number re-centres), else only those that differ.
    public func changes(at index: Int) -> Bool {
        let old = Array(from)
        let new = Array(to)
        guard old.count == new.count, index < new.count else { return true }
        return old[index] != new[index]
    }
}
