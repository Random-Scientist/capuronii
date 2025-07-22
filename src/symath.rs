use std::{
    cell::RefCell,
    fmt::Debug,
    marker::PhantomData,
    mem::take,
    num::NonZeroUsize,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Range, Sub, SubAssign},
};

pub mod backends;
pub mod numeric;

pub(crate) trait Runtime:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
    + From<Exactly>
    + Copy
{
}
impl Runtime for fsym {}

/// Computes 2^(floor(bits / 2) + 1) + 1
/// Value only changes every 2 increments, this gives a correct result for significand bitdepths both with and without the implied leading 1
const fn compute_splitter(num_significand_bits: u32) -> u32 {
    assert!(num_significand_bits > 3);
    let n = num_significand_bits / 2 + 1;
    2u32.pow(n) + 1
}
impl From<Exactly> for f64 {
    fn from(value: Exactly) -> Self {
        // 53 significand bits
        const F64_SPLITTER: f64 = compute_splitter(53) as f64;
        match value {
            Exactly::Rational(rat) => {
                let mut val = rat.frac.0 as f64 / rat.frac.1 as f64;
                if !rat.is_positive {
                    val = -val;
                }
                val
            }
            Exactly::Special(special) => match special {
                crate::symath::Special::Pi => std::f64::consts::PI,
                crate::symath::Special::Splitter => F64_SPLITTER,
            },
        }
    }
}
impl From<Exactly> for f32 {
    fn from(value: Exactly) -> Self {
        // 24 significant bits
        const F32_SPLITTER: f32 = compute_splitter(24) as f32;
        match value {
            Exactly::Rational(rat) => {
                let mut val = rat.frac.0 as f32 / rat.frac.1 as f32;
                if !rat.is_positive {
                    val = -val;
                }
                val
            }
            Exactly::Special(special) => match special {
                crate::symath::Special::Pi => std::f32::consts::PI,
                crate::symath::Special::Splitter => F32_SPLITTER,
            },
        }
    }
}
impl Runtime for f32 {}
impl Runtime for f64 {}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational64 {
    is_positive: bool,
    frac: (u64, u64),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Special {
    Pi,
    /// See [`compute_splitter`]
    Splitter,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Exactly {
    Rational(Rational64),
    Special(Special),
}

impl From<Rational64> for Exactly {
    fn from(value: Rational64) -> Self {
        Exactly::Rational(value)
    }
}
impl From<i64> for Exactly {
    fn from(value: i64) -> Self {
        Exactly::Rational(Rational64 {
            is_positive: value >= 0,
            frac: (value.unsigned_abs(), 1),
        })
    }
}
impl From<u64> for Exactly {
    fn from(value: u64) -> Self {
        Exactly::Rational(Rational64 {
            is_positive: true,
            frac: (value, 1),
        })
    }
}
impl From<Special> for Exactly {
    fn from(value: Special) -> Self {
        Self::Special(value)
    }
}
impl From<i32> for Exactly {
    fn from(value: i32) -> Self {
        i64::from(value).into()
    }
}
thread_local! {
    static CONTEXT: RefCell<SymathContext> = SymathContext::new().into();
}
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct SymHandle(NonZeroUsize, NotSend);

impl Debug for SymHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("[{}]", self.0))
    }
}
impl From<NonZeroUsize> for SymHandle {
    fn from(value: NonZeroUsize) -> Self {
        SymHandle(value, PhantomData)
    }
}
#[derive(Debug, Clone)]
enum SymathValue {
    Parameter(&'static str),
    Constant(Exactly),
    Binary(BinaryOp, SymHandle, SymHandle),
    Unary(UnaryOp, SymHandle),
    LoadLocal(NonZeroUsize),
}
#[derive(Debug, Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}
#[derive(Debug, Clone, Copy)]
enum UnaryOp {
    Neg,
    Sqrt,
}
type NotSend = PhantomData<*const ()>;
#[derive(Debug, Clone, Copy)]
enum Comparison {
    Eqal,
    GreaterEqual,
    LessEqual,
    Greater,
    Less,
}
pub struct SymathContext {
    current_func: Option<SymathCompiling>,
}
impl SymathContext {
    fn get_compiling(&mut self) -> &mut SymathCompiling {
        self.current_func
            .as_mut()
            .expect("tried to evaluate a Symath operation outside of a symath function!")
    }
}
fn with_compiling<R>(func: impl FnOnce(&mut SymathCompiling) -> R) -> R {
    CONTEXT.with(|a| func(a.borrow_mut().get_compiling()))
}
#[derive(Debug, Clone, Default)]
pub struct Block {
    inner: Vec<Statement>,
}
impl Block {
    fn new() -> Self {
        Self { inner: Vec::new() }
    }
}

#[derive(Debug, Clone)]
enum Statement {
    If {
        lhs: SymHandle,
        op: Comparison,
        rhs: SymHandle,
        accept: Block,
        reject: Block,
    },
    ExpressionRange(Range<usize>),
    Return(SymHandle),
    StoreLocal(NonZeroUsize, SymHandle),
}
#[derive(Debug, Clone)]
pub struct SymathCompiling {
    name: &'static str,
    expressions: Vec<SymathValue>,
    local_ctr: NonZeroUsize,
    statements: Block,
}
impl SymathCompiling {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            expressions: Vec::new(),
            local_ctr: const { NonZeroUsize::new(1).unwrap() },
            statements: Block::new(),
        }
    }
    fn insert(&mut self, expr: SymathValue) -> SymHandle {
        if !self
            .statements
            .inner
            .last()
            .is_some_and(|a| matches!(a, Statement::ExpressionRange(_)))
        {
            self.statements.inner.push(Statement::ExpressionRange(
                self.expressions.len()..self.expressions.len(),
            ));
        }
        let i = self.expressions.len();
        self.expressions.push(expr);
        let Statement::ExpressionRange(r) = self.statements.inner.last_mut().unwrap() else {
            unreachable!()
        };
        r.end += 1;
        NonZeroUsize::new(i + 1).unwrap().into()
    }
    fn new_constant(&mut self, val: Exactly) -> SymHandle {
        self.insert(SymathValue::Constant(val))
    }
    fn binary(&mut self, op: BinaryOp, lhs: SymHandle, rhs: SymHandle) -> SymHandle {
        self.insert(SymathValue::Binary(op, lhs, rhs))
    }
    fn unary(&mut self, op: UnaryOp, val: SymHandle) -> SymHandle {
        self.insert(SymathValue::Unary(op, val))
    }
    fn enter_block(&mut self) -> Block {
        take(&mut self.statements)
    }
    fn next_local(&mut self) -> Local {
        let l = self.local_ctr;

        self.local_ctr = l.checked_add(1).unwrap();

        Local { id: l }
    }
    fn enumerate_range(&self, r: Range<usize>) -> impl Iterator<Item = (&SymathValue, SymHandle)> {
        self.expressions[r.clone()]
            .iter()
            .zip(r.map(|e| NonZeroUsize::new(e + 1).unwrap().into()))
    }
}
impl SymathContext {
    fn new() -> Self {
        Self { current_func: None }
    }
}
impl From<Local> for fsym {
    fn from(value: Local) -> Self {
        fsym {
            handle: with_compiling(|c| c.insert(SymathValue::LoadLocal(value.id))),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Local {
    id: NonZeroUsize,
}

fn local(val: impl Into<Option<fsym>>) -> Local {
    with_compiling(move |c| {
        let loc = c.next_local();
        if let Some(v) = val.into() {
            c.statements
                .inner
                .push(Statement::StoreLocal(loc.id, v.handle));
        }
        loc
    })
}
fn uninit() -> Local {
    with_compiling(SymathCompiling::next_local)
}

impl Local {
    fn store(self, val: impl Into<fsym>) {
        with_compiling(|c| {
            c.statements
                .inner
                .push(Statement::StoreLocal(self.id, val.into().handle))
        })
    }
}

fn branch(lhs: fsym, op: Comparison, rhs: fsym, accept: impl FnOnce(), reject: impl FnOnce()) {
    let mut out = with_compiling(|c| c.enter_block());
    accept();
    let accept = with_compiling(|c| c.enter_block());
    reject();
    let reject = with_compiling(|c| c.enter_block());
    let statement = Statement::If {
        lhs: lhs.handle,
        op,
        rhs: rhs.handle,
        accept,
        reject,
    };
    out.inner.push(statement);
    with_compiling(move |c| c.statements = out)
}
fn ret(val: fsym) {
    with_compiling(|c| c.statements.inner.push(Statement::Return(val.handle)))
}

fn symath_compile(to_compile: impl Fn(), name: &'static str) -> SymathCompiling {
    CONTEXT.with(|a| a.borrow_mut().current_func = Some(SymathCompiling::new(name)));
    to_compile();
    CONTEXT
        .with(|a| a.borrow_mut().current_func.take())
        .unwrap()
}
fn input(name: &'static str) -> fsym {
    with_compiling(|c| c.insert(SymathValue::Parameter(name))).into()
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub struct fsym {
    handle: SymHandle,
}
impl From<SymHandle> for fsym {
    fn from(handle: SymHandle) -> Self {
        fsym { handle }
    }
}
impl fsym {
    fn new(val: impl Into<Exactly>) -> Self {
        val.into().into()
    }
}
impl<T: Into<Exactly>> From<T> for fsym {
    fn from(value: T) -> Self {
        with_compiling(move |c| c.new_constant(value.into())).into()
    }
}
impl Add for fsym {
    type Output = fsym;
    fn add(self, rhs: Self) -> Self::Output {
        with_compiling(|c| c.binary(BinaryOp::Add, self.handle, rhs.handle)).into()
    }
}
impl Sub for fsym {
    type Output = fsym;
    fn sub(self, rhs: Self) -> Self::Output {
        with_compiling(|c| c.binary(BinaryOp::Sub, self.handle, rhs.handle)).into()
    }
}
impl Mul for fsym {
    type Output = fsym;
    fn mul(self, rhs: Self) -> Self::Output {
        with_compiling(|c| c.binary(BinaryOp::Mul, self.handle, rhs.handle)).into()
    }
}
impl Div for fsym {
    type Output = fsym;
    fn div(self, rhs: Self) -> Self::Output {
        with_compiling(|c| c.binary(BinaryOp::Div, self.handle, rhs.handle)).into()
    }
}
impl AddAssign for fsym {
    fn add_assign(&mut self, rhs: Self) {
        *self = with_compiling(|c| c.binary(BinaryOp::Add, self.handle, rhs.handle)).into()
    }
}
impl SubAssign for fsym {
    fn sub_assign(&mut self, rhs: Self) {
        *self = with_compiling(|c| c.binary(BinaryOp::Sub, self.handle, rhs.handle)).into()
    }
}
impl MulAssign for fsym {
    fn mul_assign(&mut self, rhs: Self) {
        *self = with_compiling(|c| c.binary(BinaryOp::Mul, self.handle, rhs.handle)).into()
    }
}
impl DivAssign for fsym {
    fn div_assign(&mut self, rhs: Self) {
        *self = with_compiling(|c| c.binary(BinaryOp::Div, self.handle, rhs.handle)).into()
    }
}
impl Neg for fsym {
    type Output = fsym;

    fn neg(self) -> Self::Output {
        with_compiling(|c| c.unary(UnaryOp::Neg, self.handle)).into()
    }
}

macro_rules! symath_func {
    (
        fn $name:ident ( $( $arg:ident ),* ) {
            $( $body:tt )*
        }
    ) => {
        $crate::symath::symath_compile(
            || {
                $(
                    let $arg = $crate::symath::input(::std::stringify!($arg));
                )*
                $(
                    $body
                )*
            },
            ::std::stringify!($name),
        )

    };
}
#[cfg(test)]
mod test {
    use crate::symath::{fsym, numeric::double_single::DoubleSingle};

    #[test]
    fn test_symath() {
        let c = symath_func! {
            fn test(a, b) {
                let d = DoubleSingle { a, b };
                let b: DoubleSingle<fsym> = DoubleSingle::from_single(200.into());
                let mut s = d + b;
                s *= fsym::from(2);
                s *= s;
                let s = s / s;
            }
        };
    }
}
