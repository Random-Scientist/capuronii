use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::symath::{Exactly, Runtime, Special, fsym};

#[derive(Debug, Clone, Copy)]
pub struct DoubleSingle<Rt = fsym> {
    pub a: Rt,
    pub b: Rt,
}
#[test]
fn test() {
    dbg!(two_sum(1.0, 2.0));
    dbg!(DoubleSingle::from_sum(1.0, 2.0));
}
impl<Rt> From<(Rt, Rt)> for DoubleSingle<Rt> {
    fn from(value: (Rt, Rt)) -> Self {
        Self {
            a: value.0,
            b: value.1,
        }
    }
}
fn quick_two_sum<Rt: Runtime>(a: Rt, b: Rt) -> (Rt, Rt) {
    let s = a + b;
    (s, b - (s - a))
}
fn two_sum<Rt: Runtime>(a: Rt, b: Rt) -> (Rt, Rt) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
fn two_diff<Rt: Runtime>(a: Rt, b: Rt) -> (Rt, Rt) {
    let s = a - b;
    let bb = s - a;
    (s, (a - (s - bb)) - (b + bb))
}

fn two_prod<Rt: Runtime>(a: Rt, b: Rt) -> (Rt, Rt) {
    let p = a * b;
    let (a_hi, a_lo) = split(a);

    let (b_hi, b_lo) = split(b);

    (
        p,
        ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo,
    )
}
fn split<Rt: Runtime>(a: Rt) -> (Rt, Rt) {
    let temp = a * Rt::from(Exactly::from(Special::Splitter));
    let hi = temp - (temp - a);
    (hi, a - hi)
}

impl<Rt: Runtime> DoubleSingle<Rt> {
    pub fn from_sum(a: Rt, b: Rt) -> Self {
        two_sum(a, b).into()
    }
    pub fn from_single(v: Rt) -> Self {
        Self {
            a: v,
            b: Exactly::from(0).into(),
        }
    }
    pub fn to_single(self) -> Rt {
        self.a + self.b
    }
}

impl<Rt: Runtime> Add<Rt> for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;

    fn add(self, rhs: Rt) -> Self::Output {
        let (s1, mut s2) = two_sum(self.a, rhs);
        s2 += self.b;
        let (a, b) = quick_two_sum(s1, s2);
        DoubleSingle { a, b }
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    pub fn ieee_add(self, other: Self) -> Self {
        let (s1, s2) = two_sum(self.a, other.a);
        let (t1, t2) = two_sum(self.b, other.b);

        let (s1, s2) = quick_two_sum(s1, s2 + t1);
        quick_two_sum(s1, s2 + t2).into()
    }
    pub fn sloppy_add(self, other: Self) -> Self {
        let (s, e) = two_sum(self.a, other.a);
        quick_two_sum(s, e + self.b + other.b).into()
    }
}

impl<Rt: Runtime> Add for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;

    fn add(self, rhs: Self) -> Self::Output {
        self.ieee_add(rhs)
    }
}
impl<Rt: Runtime> AddAssign<Rt> for DoubleSingle<Rt> {
    fn add_assign(&mut self, rhs: Rt) {
        *self = *self + rhs;
    }
}
impl<Rt: Runtime> AddAssign<DoubleSingle<Rt>> for DoubleSingle<Rt> {
    fn add_assign(&mut self, rhs: DoubleSingle<Rt>) {
        *self = *self + rhs;
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    pub fn from_diff(a: Rt, b: Rt) -> Self {
        let (a, b) = two_diff(a, b);
        Self { a, b }
    }
}
impl<Rt: Runtime> Sub<Rt> for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;
    fn sub(self, rhs: Rt) -> Self::Output {
        let (s1, s2) = two_diff(self.a, rhs);
        #[allow(clippy::suspicious_arithmetic_impl)]
        quick_two_sum(s1, s2 + self.b).into()
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    fn sub_from(self, from: Rt) -> Self {
        let (s1, s2) = two_diff(from, self.a);
        quick_two_sum(s1, s2 - self.b).into()
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    pub fn sloppy_sub(self, other: Self) -> Self {
        let (s, e) = two_diff(self.a, other.a);
        quick_two_sum(s, e + self.b - other.b).into()
    }
    pub fn ieee_sub(self, other: Self) -> Self {
        let (s1, s2) = two_diff(self.a, other.a);
        let (t1, t2) = two_diff(self.b, other.b);

        let (s1, s2) = quick_two_sum(s1, s2 + t1);
        quick_two_sum(s1, s2 + t2).into()
    }
}

impl<Rt: Runtime> Sub<DoubleSingle<Rt>> for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;
    fn sub(self, rhs: DoubleSingle<Rt>) -> Self::Output {
        self.ieee_sub(rhs)
    }
}
impl<Rt: Runtime> SubAssign<Rt> for DoubleSingle<Rt> {
    fn sub_assign(&mut self, rhs: Rt) {
        *self = *self - rhs;
    }
}
impl<Rt: Runtime> SubAssign<DoubleSingle<Rt>> for DoubleSingle<Rt> {
    fn sub_assign(&mut self, rhs: DoubleSingle<Rt>) {
        *self = *self - rhs;
    }
}
impl<Rt: Runtime> Neg for DoubleSingle<Rt> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    pub fn from_prod(a: Rt, b: Rt) -> Self {
        two_prod(a, b).into()
    }
}

impl<Rt: Runtime> Mul<Rt> for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;

    fn mul(self, rhs: Rt) -> Self::Output {
        let (p1, p2) = two_prod(self.a, rhs);
        quick_two_sum(p1, p2 + (self.b * rhs)).into()
    }
}

// let temp = a * Rt::from(Exactly::from(Special::Splitter));
//     let hi = temp - (temp - a);
//     (hi, a - hi)

// fn two_prod<Rt: Runtime>(a: Rt, b: Rt) -> (Rt, Rt) {
//     let p = a * b;
//     let (a_hi, a_lo) = split(a);

//     let (b_hi, b_lo) = split(b);

//     (
//         p,
//         (
//             (a_hi * b_hi - p)
//             + a_hi * b_lo
//             + a_lo * b_hi
//         ) + a_lo * b_lo,
//     )
// }

// vec2 dsmul(vec2 dsa, vec2 dsb) {
//     vec2 dsc;
//     float c11, c21, c2, e, t1, t2;
//     float a1, a2, b1, b2, cona, conb, split = 8193.;
//     split(dsa.x)
//     cona = dsa.x * split;
//     a_hi = sub_frc(cona, sub_frc(cona, dsa.x));
//     a_lo = sub_frc(dsa.x, a_hi);
//     split(dsb.x)
//     conb = dsb.x * split;
//     b_hi = sub_frc(conb, sub_frc(conb, dsb.x));
//     b_lo = sub_frc(dsb.x, b_hi);

//     p = mul_frc(dsa.x, dsb.x);
//     p2 = add_frc(
//          mul_frc(a_lo, b_lo),
//          add_frc(
//              mul_frc(a_lo, b_hi),
//              add_frc(
//                  mul_frc(a_hi, b_lo),
//                  sub_frc(
//                      mul_frc(a_hi, b_hi),
//                      p
//                  )
//              )
//          )
//      );
//     p2 = (a_lo * b_lo) + (a_lo * b_hi + a_hi * b_lo + (a_hi * b_hi - p))

//     c2 = add_frc(mul_frc(dsa.x, dsb.y), mul_frc(dsa.y, dsb.x));

//     t1 = add_frc(p, c2);
//     e = sub_frc(t1, p);

//     t2 = (dsa.y * dsb.y + ((c2 - e) + (p - (t1 - e)))) + p2;

//     return quick_two_sum(t1, t2);
// }

impl<Rt: Runtime> Mul for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;

    fn mul(self, rhs: Self) -> Self::Output {
        let (p1, p2) = two_prod(self.a, rhs.a);
        quick_two_sum(p1, p2 + (self.a * rhs.b + self.b * rhs.a)).into()
    }
}
impl<Rt: Runtime> MulAssign<Rt> for DoubleSingle<Rt> {
    fn mul_assign(&mut self, rhs: Rt) {
        *self = *self * rhs;
    }
}
impl<Rt: Runtime> MulAssign for DoubleSingle<Rt> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    fn from_div(a: Rt, b: Rt) -> Self {
        let q1 = a / b;
        let (p1, p2) = two_prod(q1, b);
        let (s, e) = two_diff(a, p1);
        quick_two_sum(q1, (s + (e - p2)) / b).into()
    }
}
impl<Rt: Runtime> Div<Rt> for DoubleSingle<Rt> {
    type Output = DoubleSingle<Rt>;
    fn div(self, rhs: Rt) -> Self::Output {
        let q1 = self.a / rhs;
        let (p1, p2) = two_prod(q1, rhs);
        let (s, e) = two_diff(self.a, p1);
        quick_two_sum(q1, (s + ((e + self.b) - p2)) / rhs).into()
    }
}
impl<Rt: Runtime> DoubleSingle<Rt> {
    fn sloppy_div(self, other: Self) -> Self {
        let q1 = self.a / other.a;

        let r = other * q1;

        let (s1, s2) = two_diff(self.a, r.a);

        let q2 = (s1 + ((s2 - r.b) + self.b)) / other.b;

        quick_two_sum(q1, q2).into()
    }
    fn accurate_div(self, other: Self) -> Self {
        let q1 = self.a / other.a;
        let r = self - (other * q1);
        let q2 = r.a / other.a;

        let r = r - other * q2;
        let q3 = r.a / other.a;

        let ret: DoubleSingle<Rt> = quick_two_sum(q1, q2).into();

        ret + q3
    }
}
impl<Rt: Runtime> Div for DoubleSingle<Rt> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.accurate_div(rhs)
    }
}
