use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::symath::{Special, fsym};

#[derive(Clone, Copy)]
pub struct Dsym {
    pub a: fsym,
    pub b: fsym,
}
impl From<(fsym, fsym)> for Dsym {
    fn from(value: (fsym, fsym)) -> Self {
        Self {
            a: value.0,
            b: value.1,
        }
    }
}
fn quick_two_sum(a: fsym, b: fsym) -> (fsym, fsym) {
    let s = a + b;
    (s, b - (s - a))
}
fn two_sum(a: fsym, b: fsym) -> (fsym, fsym) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
fn two_diff(a: fsym, b: fsym) -> (fsym, fsym) {
    let s = a - b;
    let bb = s - a;
    (s, (a - (s - bb)) - (b + bb))
}
fn two_prod(a: fsym, b: fsym) -> (fsym, fsym) {
    let p = a * b;
    let (a_hi, a_lo) = split(a);
    let (b_hi, b_lo) = split(b);
    (
        p,
        ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo,
    )
}
fn split(a: fsym) -> (fsym, fsym) {
    let temp = a * fsym::from(Special::Splitter);
    let hi = temp - (temp - a);
    (hi, a - hi)
}

impl Dsym {
    pub fn from_sum(a: fsym, b: fsym) -> Self {
        let (a, b) = two_sum(a, b);
        Self { a, b }
    }
    pub fn from_single(v: fsym) -> Self {
        Self { a: v, b: 0.into() }
    }
}

macro_rules! two_way {
    (
     impl $trait:ident $fname:ident = |$identa:ident : $tya:ty, $identb:ident : $tyb:ty | -> $res:ty {
        $( $body:tt )*
    } ) => {
        impl $trait<$tyb> for $tya {
            type Output = $res;
            fn $fname(self, rhs: $tyb) -> Self::Output {
                let $identa = self;
                let $identb = rhs;
                $( $body )*
            }
        }
        impl $trait<$tya> for $tyb {
            type Output = $res;
            fn $fname(self, rhs: $tya) -> Self::Output {
                let $identb = self;
                let $identa = rhs;
                $( $body )*
            }
        }
    };
}
two_way! {
    impl Add add = |d: Dsym, f: fsym| -> Dsym {
        let (s1, mut s2) = two_sum(d.a, f);
        s2 += d.b;
        let (a, b) = quick_two_sum(s1, s2);
        Dsym { a, b }
    }
}
impl Dsym {
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

impl Add for Dsym {
    type Output = Dsym;

    fn add(self, rhs: Self) -> Self::Output {
        self.ieee_add(rhs)
    }
}
impl AddAssign<fsym> for Dsym {
    fn add_assign(&mut self, rhs: fsym) {
        *self = *self + rhs;
    }
}
impl AddAssign<Dsym> for Dsym {
    fn add_assign(&mut self, rhs: Dsym) {
        *self = *self + rhs;
    }
}
impl Dsym {
    pub fn from_diff(a: fsym, b: fsym) -> Self {
        let (a, b) = two_diff(a, b);
        Self { a, b }
    }
}
impl Sub<fsym> for Dsym {
    type Output = Dsym;
    fn sub(self, rhs: fsym) -> Self::Output {
        let (s1, s2) = two_diff(self.a, rhs);
        #[allow(clippy::suspicious_arithmetic_impl)]
        quick_two_sum(s1, s2 + self.b).into()
    }
}
impl Sub<Dsym> for fsym {
    type Output = Dsym;

    fn sub(self, rhs: Dsym) -> Self::Output {
        let (s1, s2) = two_diff(self, rhs.a);
        quick_two_sum(s1, s2 - rhs.b).into()
    }
}
impl Dsym {
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

impl Sub<Dsym> for Dsym {
    type Output = Dsym;
    fn sub(self, rhs: Dsym) -> Self::Output {
        self.ieee_sub(rhs)
    }
}
impl SubAssign<fsym> for Dsym {
    fn sub_assign(&mut self, rhs: fsym) {
        *self = *self - rhs;
    }
}
impl SubAssign<Dsym> for Dsym {
    fn sub_assign(&mut self, rhs: Dsym) {
        *self = *self - rhs;
    }
}
impl Neg for Dsym {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }
}
impl Dsym {
    pub fn from_prod(a: fsym, b: fsym) -> Self {
        two_prod(a, b).into()
    }
}

two_way! {
    impl Mul mul = |d: Dsym, f: fsym| -> Dsym {
        let (p1, p2) = two_prod(d.a, f);
        quick_two_sum(p1, p2 + (d.b * f)).into()
    }
}
impl Mul for Dsym {
    type Output = Dsym;

    fn mul(self, rhs: Self) -> Self::Output {
        let (p1, p2) = two_prod(self.a, rhs.a);
        quick_two_sum(p1, p2 + (self.a * rhs.b + self.b * rhs.a)).into()
    }
}
impl MulAssign<fsym> for Dsym {
    fn mul_assign(&mut self, rhs: fsym) {
        *self = *self * rhs;
    }
}
impl MulAssign for Dsym {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Dsym {
    fn from_div(a: fsym, b: fsym) -> Self {
        let q1 = a / b;
        let (p1, p2) = two_prod(q1, b);
        let (s, e) = two_diff(a, p1);
        quick_two_sum(q1, (s + (e - p2)) / b).into()
    }
}
impl Div<fsym> for Dsym {
    type Output = Dsym;
    fn div(self, rhs: fsym) -> Self::Output {
        let q1 = self.a / rhs;
        let (p1, p2) = two_prod(q1, rhs);
        let (s, e) = two_diff(self.a, p1);
        quick_two_sum(q1, (s + ((e + self.b) - p2)) / rhs).into()
    }
}
impl Dsym {
    fn sloppy_div(self, other: Self) -> Self {
        let q1 = self.a / other.a;

        let r = other * q1;

        let (s1, s2) = two_diff(self.a, r.a);

        let q2 = (s1 + ((s2 - r.b) + self.b)) / other.b;

        quick_two_sum(q1, q2).into()
    }
    fn accurate_div(self, other: Self) -> Self {
        let q1 = self.a / other.a;
        let r = self - (q1 * other);
        let q2 = r.a / other.a;

        let r = r - q2 * other;
        let q3 = r.a / other.a;

        let ret: Dsym = quick_two_sum(q1, q2).into();

        ret + q3
    }
}
impl Div for Dsym {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.accurate_div(rhs)
    }
}
impl Div<Dsym> for fsym {
    type Output = Dsym;
    fn div(self, rhs: Dsym) -> Self::Output {
        Dsym::from_single(self) / rhs
    }
}
