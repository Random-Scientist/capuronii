use naga::{Expression, Handle};

use crate::{Compiler, function::CompilingFunction};
#[derive(Debug, Clone, Copy)]
enum MathMode {
    /// Assume non-NaN and non-inf
    Unchecked,
    /// Checking state
    Checked(NanCheck),
}
/// Represents a NaN check done at an exact point in expression evaluation
#[derive(Debug, Clone, Copy)]
pub struct NanCheck {
    // handle to bool
    is_nan: Handle<Expression>,
    // handle to u32 encoded expression id of the expression where this NaN was generated
    source: Handle<Expression>,
}
impl MathMode {
    fn checked(self) -> Option<NanCheck> {
        match self {
            MathMode::Unchecked => None,
            MathMode::Checked(handle) => Some(handle),
        }
    }
    fn is_nan(self, func: &CompilingFunction) -> Handle<Expression> {
        match self {
            MathMode::Unchecked => func.constants.false_bool,
            MathMode::Checked(handle) => handle.is_nan,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Float32 {
    /// value of this [`Float32`]. Do not use directly without a NaN-check if [`Float32::mode`] is [`MathMode::Checked`]
    value: Handle<Expression>,
    /// The checking mode this value was produced under
    mode: MathMode,
}
impl Float32 {
    /// Gets the (possibly-NaN) bitwise representation of this [`Float32`] as a u32
    fn bits(self, func: &mut CompilingFunction, ctx: &Compiler) -> Handle<Expression> {
        let cast = func.bitcast_to_u32(self.value);
        if let MathMode::Checked(c) = self.mode
            && !ctx.config.assume_numbers_finite
        {
            let mut nanval = func.constants.canonical_nan;
            if ctx.config.do_nan_tracking {
                nanval = func.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::InclusiveOr,
                    left: nanval,
                    right: c.source,
                });
            }
            func.add_unspanned(Expression::Select {
                condition: c.is_nan,
                accept: nanval,
                reject: cast,
            })
        } else {
            cast
        }
    }
    fn from_bits(func: &mut CompilingFunction, ctx: &Compiler, bits: Handle<Expression>) -> Self {
        let mode = if ctx.config.assume_numbers_finite {
            MathMode::Unchecked
        } else {
            let and_nanbase = func.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::And,
                left: bits,
                right: func.constants.canonical_nan,
            });
            let is_nan = func.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Equal,
                left: and_nanbase,
                right: func.constants.canonical_nan,
            });
            let source = if ctx.config.do_nan_tracking {
                func.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::ExclusiveOr,
                    left: bits,
                    right: func.constants.canonical_nan,
                })
            } else {
                func.constants.zero_u32
            };
            MathMode::Checked(NanCheck { is_nan, source })
        };
        let value = func.bitcast_to_float(bits);
        Self { value, mode }
    }
}

impl CompilingFunction {
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
            arg1: Some(self.constants.one_f32),
            arg2: Some(self.constants.max_f32),
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
            right: self.constants.one_f32,
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
            right: self.constants.log2_max_f32,
        })
    }
    pub(crate) fn propagate_nan<const OPERANDS: usize>(
        &mut self,
        ctx: &Compiler,
        operands: [NanCheck; OPERANDS],
        result_is_nan: Handle<Expression>,
        result_source_id: u32,
    ) -> NanCheck {
        const { assert!(OPERANDS > 0) }

        let base = operands[0];

        let operand_check = operands[1..].iter().fold(base, |accum, b| {
            let is_nan = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left: accum.is_nan,
                right: b.is_nan,
            });
            let source = if ctx.config.do_nan_tracking {
                self.add_unspanned(Expression::Select {
                    condition: b.is_nan,
                    accept: b.source,
                    reject: accum.source,
                })
            } else {
                self.constants.zero_u32
            };
            NanCheck { is_nan, source }
        });
        let is_nan = self.add_unspanned(Expression::Select {
            condition: operand_check.is_nan,
            accept: self.constants.true_bool,
            reject: result_is_nan,
        });
        let source = if ctx.config.do_nan_tracking {
            let this_id =
                self.add_preemit(Expression::Literal(naga::Literal::U32(result_source_id)));
            self.add_unspanned(Expression::Select {
                condition: result_is_nan,
                accept: this_id,
                reject: operand_check.source,
            })
        } else {
            self.constants.zero_u32
        };

        NanCheck { is_nan, source }
    }
    pub(crate) fn add(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        let (lhs, rhs) = (lhsf.value, rhsf.value);
        let value = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: lhs,
            right: rhs,
        });

        // NaN checking is only possible for finite inputs
        let mode = if let (Some(lhsnan), Some(rhsnan)) = (lhsf.mode.checked(), rhsf.mode.checked())
        {
            let result_is_nan = self.check_sum(lhs, rhs);

            let id = value.index() as u32;
            MathMode::Checked(self.propagate_nan(ctx, [lhsnan, rhsnan], result_is_nan, id))
        } else {
            MathMode::Unchecked
        };
        Float32 { value, mode }
    }
    pub(crate) fn sub(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        let (lhs, rhs) = (lhsf.value, rhsf.value);
        let value = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Subtract,
            left: lhs,
            right: rhs,
        });
        // NaN checking is only possible for finite inputs
        let mode = if let (Some(lhsnan), Some(rhsnan)) = (lhsf.mode.checked(), rhsf.mode.checked())
        {
            // a - b = a + (-b)
            let rhs = self.add_unspanned(Expression::Unary {
                op: naga::UnaryOperator::Negate,
                expr: rhs,
            });
            let result_is_nan = self.check_sum(lhs, rhs);

            let id = value.index() as u32;
            MathMode::Checked(self.propagate_nan(ctx, [lhsnan, rhsnan], result_is_nan, id))
        } else {
            MathMode::Unchecked
        };
        Float32 { value, mode }
    }
    pub(crate) fn mul(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        let (lhs, rhs) = (lhsf.value, rhsf.value);
        let value = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: lhs,
            right: rhs,
        });
        let mode = if let (Some(lhsnan), Some(rhsnan)) = (lhsf.mode.checked(), rhsf.mode.checked())
        {
            let labs = self.abs_log2(lhs);
            let rabs = self.abs_log2(rhs);
            let f = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: labs,
                right: rabs,
            });
            let result_is_nan = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Greater,
                left: f,
                right: self.constants.log2_max_f32,
            });

            let id = value.index() as u32;
            MathMode::Checked(self.propagate_nan(ctx, [lhsnan, rhsnan], result_is_nan, id))
        } else {
            MathMode::Unchecked
        };
        Float32 { value, mode }
    }
    pub(crate) fn div(&mut self, ctx: &Compiler, lhsf: Float32, rhsf: Float32) -> Float32 {
        let (lhs, rhs) = (lhsf.value, rhsf.value);
        let value = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: lhs,
            right: rhs,
        });

        let mode = if let (Some(lhsnan), Some(rhsnan)) = (lhsf.mode.checked(), rhsf.mode.checked())
        {
            let rzero = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Equal,
                left: rhs,
                right: self.constants.zero_f32,
            });
            let logl = self.abs_log2(lhs);
            let logr = self.abs_log2(rhs);
            let f = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Subtract,
                left: logl,
                right: logr,
            });
            let nonzero_result_nan = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Greater,
                left: f,
                right: self.constants.log2_max_f32,
            });
            let result_is_nan = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::LogicalOr,
                left: nonzero_result_nan,
                right: rzero,
            });

            let id = value.index() as u32;
            MathMode::Checked(self.propagate_nan(ctx, [lhsnan, rhsnan], result_is_nan, id))
        } else {
            MathMode::Unchecked
        };
        Float32 { value, mode }
    }
}
