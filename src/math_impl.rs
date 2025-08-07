use naga::{Expression, Handle, MathFunction};
use parse::{ast::ComparisonOperator, type_checker};

use crate::{Compiler, ScalarValue, function::CompilingFunction};

pub enum NanCheckMode {
    /// Skip generating NaN checking code entirely. The result of evaluating a expression that produces any non-finite value is undefined.
    AssumeFinite,
    /// Assume that the backing WebGPU implementation supports NaN as an implementation detail,
    /// and implement NaN checking with a bitwise inspection of the value for NaN bit patterns.
    AssumeSupported,
    /// Emulate NaN checking in software.
    /// If [`do_overflow_check`](NanCheckMode::PortabilityMode::do_overflow_check) is `true`, this checking includes logarithmic magnitude comparisons to
    /// detect when numeric operations overflow, at a significant performance cost.
    /// Always performs domain checking e.g. division by zero, even roots of negative numbers... etc.
    PortabilityMode { do_overflow_check: bool },
}
pub struct NumericConfig {
    /// Whether to propagate the source expressions of NaNs
    /// has no effect if [`nan_check_mode`](NumericConfig::nan_check_mode) is [`AssumeFinite`](NanCheckMode::AssumeFinite)
    ///
    /// Note: may significantly reduce performance of the generated code for NaN propagation, depending on the nan checking configuration.
    pub do_nan_tracking: bool,
    /// How and when NaN checks are performed on arithmetic operations
    pub nan_check_mode: NanCheckMode,
}
impl Default for NumericConfig {
    fn default() -> Self {
        // default to the most conservative configuration
        Self {
            do_nan_tracking: false,
            nan_check_mode: NanCheckMode::PortabilityMode {
                do_overflow_check: true,
            },
        }
    }
}
impl NumericConfig {
    pub(crate) fn assume_floats_finite(&self) -> bool {
        matches!(self.nan_check_mode, NanCheckMode::AssumeFinite)
    }
    pub(crate) fn assume_nan_supported(&self) -> bool {
        matches!(self.nan_check_mode, NanCheckMode::AssumeSupported)
    }
    pub(crate) fn do_overflow_nan_checks(&self) -> bool {
        matches!(
            self.nan_check_mode,
            NanCheckMode::PortabilityMode {
                do_overflow_check: true
            }
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum Metadata {
    /// Assume non-NaN and non-inf
    AssumeFinite,
    /// Checking state
    Tracked(Option<NanCheck>),
}
/// Represents a NaN check done at an exact point in expression evaluation
#[derive(Debug, Clone, Copy)]
pub struct NanCheck {
    // handle to bool
    is_nan: Handle<Expression>,
    // handle to u32 encoded expression id of the expression where this NaN was generated
    source: Option<Handle<Expression>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Float32 {
    /// value of this [`Float32`]. Do not use directly without a NaN-check if [`Float32::mode`] is [`MathMode::Checked`]
    value: Handle<Expression>,
    /// The checking mode this value was produced under, as well as metadata containing tracking information
    mode: Metadata,
}
impl Float32 {
    pub(crate) fn new_assume_finite(value: Handle<Expression>) -> Self {
        Self {
            value,
            mode: Metadata::AssumeFinite,
        }
    }
    fn is_assume_finite(&self) -> bool {
        matches!(self.mode, Metadata::AssumeFinite)
    }
    fn get_or_materialize_check(&mut self, func: &mut CompilingFunction) -> Handle<Expression> {
        match &mut self.mode {
            Metadata::AssumeFinite => {
                // unconditionally assume finite, cannot track source without check
                func.consts.false_bool
            }
            Metadata::Tracked(nan_check) => {
                let c = nan_check.get_or_insert_with(|| {
                    let bits = func.bitcast_to_u32(self.value);
                    let is_nan = func.float_bits_are_nan(bits);
                    NanCheck {
                        is_nan,
                        source: None,
                    }
                });
                c.is_nan
            }
        }
    }
    fn get_or_materialize_source(
        &mut self,
        func: &mut CompilingFunction,
    ) -> Option<Handle<Expression>> {
        // ensure we have a nan check if we need one
        let _ = self.get_or_materialize_check(func);
        match &mut self.mode {
            Metadata::AssumeFinite => None,
            Metadata::Tracked(nan_check) => {
                let c = nan_check.as_mut().unwrap();
                Some(*c.source.get_or_insert_with(|| {
                    let bits = func.bitcast_to_u32(self.value);
                    func.float_bits_nan_payload(bits)
                }))
            }
        }
    }
    /// Gets the (possibly-NaN) bitwise representation of this [`Float32`] as a u32
    pub(crate) fn bits(
        mut self,
        func: &mut CompilingFunction,
        ctx: &Compiler,
    ) -> Handle<Expression> {
        let cast = func.bitcast_to_u32(self.value);
        // manually propagate NaN if the compiler:
        // is not configured to globally assume finite
        // is not configured to assume the implementation supports representing NaN values
        // and the value is not assumed to be finite
        if !ctx.config.numeric.assume_floats_finite()
            && !ctx.config.numeric.assume_nan_supported()
            && !self.is_assume_finite()
        {
            let mut nanval = func.consts.canonical_nan;
            if ctx.config.numeric.do_nan_tracking {
                let source = self.get_or_materialize_source(func).unwrap();
                nanval = func.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::InclusiveOr,
                    left: nanval,
                    right: source,
                });
            }
            let condition = self.get_or_materialize_check(func);
            func.add_unspanned(Expression::Select {
                condition,
                accept: nanval,
                reject: cast,
            })
        } else {
            cast
        }
    }

    pub(crate) fn from_bits(
        func: &mut CompilingFunction,
        ctx: &Compiler,
        bits: Handle<Expression>,
    ) -> Self {
        let value = func.bitcast_to_float(bits);
        let mode = if ctx.config.numeric.assume_floats_finite() {
            Metadata::AssumeFinite
        } else {
            Metadata::Tracked(if !ctx.config.numeric.assume_nan_supported() {
                // if the implementation can't represent NaN values precheck while we still have an integer
                let is_nan = func.float_bits_are_nan(bits);
                let source = if ctx.config.numeric.do_nan_tracking {
                    Some(func.float_bits_nan_payload(bits))
                } else {
                    None
                };
                Some(NanCheck { is_nan, source })
            } else {
                // else we can later emit the checks lazily via get_or_materialize_check/source
                None
            })
        };

        Self { value, mode }
    }
}

impl CompilingFunction {
    // returns bool
    pub(crate) fn cmp(
        &mut self,
        ctx: &Compiler,
        comp: ComparisonOperator,
        mut left: Float32,
        mut right: Float32,
    ) -> Handle<Expression> {
        let mut cond = self.add_unspanned(naga::Expression::Binary {
            op: match comp {
                type_checker::ComparisonOperator::Equal => naga::BinaryOperator::Equal,
                type_checker::ComparisonOperator::Less => naga::BinaryOperator::Less,
                type_checker::ComparisonOperator::LessEqual => naga::BinaryOperator::LessEqual,
                type_checker::ComparisonOperator::Greater => naga::BinaryOperator::Greater,
                type_checker::ComparisonOperator::GreaterEqual => {
                    naga::BinaryOperator::GreaterEqual
                }
            },
            left: left.value,
            right: right.value,
        });
        let left = left.get_or_materialize_check(self);
        let right = right.get_or_materialize_check(self);
        if !ctx.config.numeric.assume_floats_finite() {
            let nanck = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left,
                right,
            });
            cond = self.add_unspanned(Expression::Select {
                condition: nanck,
                accept: self.consts.false_bool,
                reject: cond,
            });
        }
        cond
    }
    pub(crate) fn new_literal(&mut self, val: f32) -> ScalarValue {
        ScalarValue::Number(if val.is_nan() || val.is_infinite() {
            Float32 {
                value: self.consts.one_f32,
                mode: Metadata::Tracked(Some(NanCheck {
                    is_nan: self.consts.true_bool,
                    source: None,
                })),
            }
        } else {
            Float32 {
                value: self.add_preemit(Expression::Literal(naga::Literal::F32(val))),
                mode: Metadata::AssumeFinite,
            }
        })
    }
    fn float_bits_nan_payload(&mut self, bits: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::ExclusiveOr,
            left: bits,
            right: self.consts.canonical_nan,
        })
    }
    fn float_bits_are_nan(&mut self, bits: Handle<Expression>) -> Handle<Expression> {
        let and_nanbase = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::And,
            left: bits,
            right: self.consts.canonical_nan,
        });
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Equal,
            left: and_nanbase,
            right: self.consts.canonical_nan,
        })
    }
    fn abs_log2(&mut self, float: Handle<Expression>) -> Handle<Expression> {
        // take the absolute value; we only care about extracting magnitude information
        let abs = self.raw_abs(float);
        // clamp abs(x) to 1.., which prevents causing NaN by log(-inf..=0)
        let abs = self.clamp_1_to_max(abs);
        self.raw_log2(abs)
    }
    fn clamp_1_to_max(&mut self, float: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::Math {
            fun: naga::MathFunction::Clamp,
            arg: float,
            arg1: Some(self.consts.one_f32),
            arg2: Some(self.consts.max_f32),
            arg3: None,
        })
    }
    fn raw_log2(&mut self, float: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::Math {
            fun: naga::MathFunction::Log2,
            arg: float,
            arg1: None,
            arg2: None,
            arg3: None,
        })
    }
    fn raw_abs(&mut self, float: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::Math {
            fun: naga::MathFunction::Abs,
            arg: float,
            arg1: None,
            arg2: None,
            arg3: None,
        })
    }
    /// Computes `log2(lhs + rhs) > log2(f32::MAX)`. The logarithm of the sum is evaluated using the identity:
    ///
    /// `log(a+b) = log(a) + log(1 + b/a)`
    ///
    /// Further, because addition is commutative, we may select a and b
    /// such that `b/a` is small so as to avoid causing an overflow in the division step
    fn check_sum(
        &mut self,
        lhs: Handle<Expression>,
        rhs: Handle<Expression>,
    ) -> Handle<Expression> {
        let labs = self.raw_abs(lhs);
        let rabs = self.raw_abs(rhs);

        let test = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Greater,
            left: labs,
            right: rabs,
        });

        let max_mag = self.add_unspanned(Expression::Select {
            condition: test,
            accept: lhs,
            reject: rhs,
        });

        let min_mag = self.add_unspanned(Expression::Select {
            condition: test,
            accept: rhs,
            reject: lhs,
        });
        let max_abs = self.add_unspanned(Expression::Select {
            condition: test,
            accept: labs,
            reject: rabs,
        });

        // clamp abs(max_mag) to 1.. and take the log
        let a_clamped = self.clamp_1_to_max(max_abs);

        let log_a = self.raw_log2(a_clamped);

        // compute log2( clamp( abs( 1 + min_mag/max_mag ), 1, f32::MAX ) )
        let b_a = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Divide,
            left: max_mag,
            right: min_mag,
        });
        let b_a_1 = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: b_a,
            right: self.consts.one_f32,
        });
        let log_b_a_1 = self.abs_log2(b_a_1);
        let final_mag = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: log_a,
            right: log_b_a_1,
        });
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Greater,
            left: final_mag,
            right: self.consts.log2_max_f32,
        })
    }
    pub(crate) fn sw_propagate_nan<const OPERANDS: usize>(
        &mut self,
        ctx: &Compiler,
        operands: &mut [Float32; OPERANDS],
        result_is_nan: Handle<Expression>,
        result_source_id: Option<u32>,
    ) -> NanCheck {
        const { assert!(OPERANDS > 0) }
        // infallible due to const assertion
        let base = &mut operands[0];

        let _ = base.get_or_materialize_check(self);
        let _ = base.get_or_materialize_source(self);
        // lift AssumeFinite values into tracked values (tracking meta is required for sw propagation)
        let Metadata::Tracked(Some(base)) = base.mode else {
            return NanCheck {
                is_nan: self.consts.false_bool,
                source: None,
            };
        };

        let operand_check = operands[1..].iter_mut().fold(base, |mut accum, b| {
            let b_is_nan = b.get_or_materialize_check(self);
            let is_nan = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left: accum.is_nan,
                right: b_is_nan,
            });
            accum.is_nan = is_nan;
            if ctx.config.numeric.do_nan_tracking {
                // get or zero source value for each operand
                let s = b
                    .get_or_materialize_source(self)
                    .unwrap_or(self.consts.zero_u32);

                let r = accum.source.get_or_insert(self.consts.zero_u32);
                *r = self.add_unspanned(Expression::Select {
                    condition: b_is_nan,
                    accept: s,
                    reject: *r,
                });
            }
            accum
        });

        let is_nan = self.add_unspanned(Expression::Select {
            condition: operand_check.is_nan,
            accept: self.consts.true_bool,
            reject: result_is_nan,
        });
        let source = if let Some(result_source_id) = result_source_id
            && ctx.config.numeric.do_nan_tracking
        {
            let this_id =
                self.add_preemit(Expression::Literal(naga::Literal::U32(result_source_id)));
            Some(self.add_unspanned(Expression::Select {
                condition: result_is_nan,
                accept: this_id,
                // operand check is guaranteed to have initialized source
                reject: operand_check.source.unwrap(),
            }))
        } else {
            None
        };

        NanCheck { is_nan, source }
    }
    pub(crate) fn fp_op<const OPERANDS: usize>(
        &mut self,
        ctx: &Compiler,
        mut args: [Float32; OPERANDS],
        compute_res: impl FnOnce(
            &mut CompilingFunction,
            [Handle<Expression>; OPERANDS],
        ) -> Handle<Expression>,
        res_is_nan: impl FnOnce(
            &mut CompilingFunction,
            &mut [Float32; OPERANDS],
            &NumericConfig,
        ) -> Handle<Expression>,
    ) -> Float32 {
        let value = compute_res(self, args.map(|a| a.value));
        if ctx.config.numeric.assume_floats_finite() {
            Float32 {
                value,
                mode: Metadata::AssumeFinite,
            }
        } else {
            let result_is_nan = res_is_nan(self, &mut args, &ctx.config.numeric);
            Float32 {
                value,
                // need to emulate propagation
                mode: Metadata::Tracked(
                    if !ctx.config.numeric.assume_nan_supported()
                        || ctx.config.numeric.do_nan_tracking
                    {
                        Some(self.sw_propagate_nan(
                            ctx,
                            &mut args,
                            result_is_nan,
                            Some(value.index() as u32),
                        ))
                    } else {
                        None
                    },
                ),
            }
        }
    }
    fn non_nan_unary_fp_op(
        &mut self,
        ctx: &Compiler,
        val: Float32,
        imp: impl FnOnce(&mut CompilingFunction, Handle<Expression>) -> Handle<Expression>,
    ) -> Float32 {
        self.fp_op(
            ctx,
            [val],
            move |func, [val]| imp(func, val),
            |f, _, _| f.consts.false_bool,
        )
    }
    pub(crate) fn add(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [lhsf, rhsf],
            |c, [lhs, rhs]| {
                c.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: lhs,
                    right: rhs,
                })
            },
            |func, [lhs, rhs], config| {
                if config.do_overflow_nan_checks() {
                    func.check_sum(lhs.value, rhs.value)
                } else {
                    // domain: R
                    func.consts.false_bool
                }
            },
        )
    }
    pub(crate) fn sub(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [lhsf, rhsf],
            |func, [lhs, rhs]| {
                func.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: lhs,
                    right: rhs,
                })
            },
            |func, [lhs, rhs], config| {
                if config.do_overflow_nan_checks() {
                    // a - b = a + (-b)
                    let rhs = func.add_unspanned(Expression::Unary {
                        op: naga::UnaryOperator::Negate,
                        expr: rhs.value,
                    });
                    func.check_sum(lhs.value, rhs)
                } else {
                    // domain: R
                    func.consts.false_bool
                }
            },
        )
    }
    pub(crate) fn mul(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [lhsf, rhsf],
            |c, [lhs, rhs]| {
                c.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    left: lhs,
                    right: rhs,
                })
            },
            |func, [lhs, rhs], config| {
                if config.do_overflow_nan_checks() {
                    let labs = func.abs_log2(lhs.value);
                    let rabs = func.abs_log2(rhs.value);
                    let f = func.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::Add,
                        left: labs,
                        right: rabs,
                    });
                    func.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::Greater,
                        left: f,
                        right: func.consts.log2_max_f32,
                    })
                } else {
                    // domain: R
                    func.consts.false_bool
                }
            },
        )
    }
    pub(crate) fn div(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [lhsf, rhsf],
            |c, [lhs, rhs]| {
                c.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Divide,
                    left: lhs,
                    right: rhs,
                })
            },
            |func, [lhs, rhs], config| {
                let domain_check = func.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Equal,
                    left: rhs.value,
                    right: func.consts.zero_f32,
                });
                if config.do_overflow_nan_checks() {
                    let labs = func.abs_log2(lhs.value);
                    let rabs = func.abs_log2(rhs.value);
                    let f = func.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::Subtract,
                        left: labs,
                        right: rabs,
                    });
                    let overflow = func.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::Greater,
                        left: f,
                        right: func.consts.log2_max_f32,
                    });
                    func.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::LogicalOr,
                        left: overflow,
                        right: domain_check,
                    })
                } else {
                    // domain: a in R s.t. a != 0
                    domain_check
                }
            },
        )
    }
    pub(crate) fn negate(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [val],
            |f, [v]| {
                f.add_unspanned(Expression::Unary {
                    op: naga::UnaryOperator::Negate,
                    expr: v,
                })
            },
            // negation of a finite value is a finite value
            |f, _, _| f.consts.false_bool,
        )
    }
    pub(crate) fn sqrt(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [val],
            |func, [v]| {
                func.add_unspanned(Expression::Math {
                    fun: naga::MathFunction::Sqrt,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            },
            |f, [v], _| {
                // Sqrt strictly shrinks the magnitude of the operand
                // domain: R+ | 0
                f.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Less,
                    left: v.value,
                    right: f.consts.zero_f32,
                })
            },
        )
    }
    pub(crate) fn ln(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [val],
            |func, [v]| {
                func.add_unspanned(Expression::Math {
                    fun: naga::MathFunction::Log,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            },
            |f, [v], _| {
                // ln strictly contracts the magnitude of the operand if operand >= 1.
                // If the operand is less than one, the magnitude may increase
                // but may not be greater than 15.9 as |ln(f32::EPSILON)| ~= 15.94,
                // so no magnitude checking is required

                // domain: R+
                f.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::LessEqual,
                    left: v.value,
                    right: f.consts.zero_f32,
                })
            },
        )
    }
    pub(crate) fn exp(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.fp_op(
            ctx,
            [val],
            |func, [val]| {
                func.add_unspanned(Expression::Math {
                    fun: naga::MathFunction::Exp,
                    arg: val,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            },
            |func, [arg], c| {
                if c.do_overflow_nan_checks() {
                    // we are interested in detecting exp(x) > f32::MAX, but computing this will cause +inf.
                    // we simply manipulate this inequality into the form x > ln(f32::MAX), precomputing the rhs
                    func.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::Greater,
                        left: arg.value,
                        right: func.consts.ln_max_f32,
                    })
                } else {
                    func.consts.false_bool
                }

                // domain: R
            },
        )
    }
    //TODO erf
    fn nnan_un_math_func(&mut self, ctx: &Compiler, val: Float32, fun: MathFunction) -> Float32 {
        self.non_nan_unary_fp_op(ctx, val, |func, val| {
            func.add_unspanned(Expression::Math {
                fun,
                arg: val,
                arg1: None,
                arg2: None,
                arg3: None,
            })
        })
    }
    pub(crate) fn sin(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.nnan_un_math_func(ctx, val, MathFunction::Sin)
    }
    pub(crate) fn cos(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.nnan_un_math_func(ctx, val, MathFunction::Cos)
    }
    // TODO hyperbolic trig inverse trig
    pub(crate) fn abs(&mut self, ctx: &Compiler, val: Float32) -> Float32 {
        self.nnan_un_math_func(ctx, val, MathFunction::Abs)
    }
}
