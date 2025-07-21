use std::{
    collections::HashMap,
    fmt::{self, Display, Write},
};

use crate::symath::{BinaryOp, Block, Exactly, SymHandle, SymathCompiling, SymathValue};
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F32,
    F64,
}
impl Precision {
    fn num_frac_bits(self) -> u32 {
        match self {
            Precision::F32 => 24,
            Precision::F64 => 52,
        }
    }
}
impl Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::F32 => f.write_str("f32"),
            Precision::F64 => f.write_str("f64"),
        }
    }
}
pub struct Config {
    prec: Precision,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            prec: Precision::F64,
        }
    }
}
impl Exactly {
    fn write_for_prec(&self, p: Precision, w: &mut impl fmt::Write) {
        match self {
            Exactly::Rational(rat) => {
                let mut val = rat.frac.0 as f64 / rat.frac.1 as f64;
                if !rat.is_positive {
                    val = -val;
                }
                write!(w, "{val}{p}").unwrap();
            }
            Exactly::Special(special) => match special {
                crate::symath::Special::Pi => write!(w, "{p}::PI").unwrap(),
                crate::symath::Special::Splitter => {
                    write!(w, "{}{p}", 2.0f64.powi(p.num_frac_bits() as i32 / 2)).unwrap();
                }
            },
        }
    }
}
impl Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        })
    }
}
impl Display for SymHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("_e{}", self.0.get()))
    }
}
pub fn compile_to_rust(func: &SymathCompiling, config: Config) -> String {
    // prepass

    let mut output = format!("fn {}(", func.name);
    func.expressions
        .iter()
        .filter_map(|a| match a {
            SymathValue::Parameter(s) => Some(*s),
            _ => None,
        })
        .for_each(|p| write!(&mut output, "{p}: {},", config.prec).unwrap());

    let p = output.pop();
    if let Some(p) = p {
        if p != ',' {
            output.push(p);
        }
    }
    output.push(')');

    let mat_expr = |val: &SymathValue, o: &mut String| match val {
        SymathValue::Parameter(c) => o.push_str(c),
        SymathValue::Constant(exactly) => exactly.write_for_prec(config.prec, o),
        SymathValue::Binary(binary_op, sym_handle, sym_handle1) => {
            write!(o, "({sym_handle} {binary_op} {sym_handle1})").unwrap();
        }
        SymathValue::Unary(unary_op, sym_handle) => todo!(),
        SymathValue::LoadLocal(non_zero) => todo!(),
    };

    let mat_block = |block: &Block, o: &mut String| {
        o.push_str("{\n");
        for s in block.inner.iter() {
            match s {
                crate::symath::Statement::If {
                    lhs,
                    op,
                    rhs,
                    accept,
                    reject,
                } => todo!(),
                crate::symath::Statement::ExpressionRange(r) => {
                    for (val, handle) in func.enumerate_range(r.clone()) {
                        write!(o, "let {handle} = ").unwrap();
                        mat_expr(val, o);

                        writeln!(o, ";").unwrap();
                    }
                }
                crate::symath::Statement::Return(sym_handle) => {
                    writeln!(o, "return {sym_handle};").unwrap();
                }
                crate::symath::Statement::StoreLocal(non_zero, sym_handle) => todo!(),
            }
        }
        o.push_str("}\n");
    };

    mat_block(&func.statements, &mut output);
    output
}
