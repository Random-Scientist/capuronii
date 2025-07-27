use naga::{
    AddressSpace, Arena, GlobalVariable, Handle, MathFunction, Module, ResourceBinding,
    StorageAccess, UniqueArena,
};
use parse::type_checker::{self, BaseType, BuiltIn, TypedExpression};
use std::collections::HashMap;
use typed_index_collections::{TiVec, ti_vec};

use crate::{
    function::CompilingFunction,
    listdef::{
        Filter, Join, LazyBroadcast, LazyComprehension, LazyStaticList, ListDef, MaterializedList,
        Select, StackList, UntypedListDef,
    },
};
mod alloc;
mod function;
mod listdef;
mod math_impl;
mod symath;
#[cfg(test)]
mod test;

struct CompilerConfig {
    /// Maximum size of the arena allocated for each invocation in intervals of 4 bytes (e.g. setting this value to 10 implies a >=40 byte long backing heap buffer)
    heap_per_invocation: u32,
    /// Map of [`Identifier`](parse::type_checker::Expression::Identifier)s to override with a dynamic GPU value
    io_map: Vec<(usize, GpuInput)>,
    /// Whether to assume that all numbers produced by the expression are finite.
    /// This falls back to the WGSL semantics; when an operation would return ±inf
    /// or NaN under IEEE it is permitted to return an arbitrary value instead.
    /// This allows us to generate **much** faster code because we don't have to emulate NaN checking in software
    assume_numbers_finite: bool,
    /// Whether to "NaN box" and propagate the source expressions of NaNs
    /// Only affects generated code when [`CompilerConfig::assume_numbers_finite`] is `false`
    /// Note: may further reduce performance of the generated code for NaN propagation.
    do_nan_tracking: bool,
}
impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            heap_per_invocation: 20000,
            io_map: Vec::new(),
            do_nan_tracking: false,
            assume_numbers_finite: false,
        }
    }
}

struct TyContext {
    u32: Handle<naga::Type>,
    f32: Handle<naga::Type>,
    point: Handle<naga::Type>,
    bool: Handle<naga::Type>,
    uvec3: Handle<naga::Type>,
}

struct Compiler {
    global_assignments: HashMap<usize, type_checker::Assignment>,
    module: Module,
    types: TyContext,
    constant_buffer: Handle<GlobalVariable>,
    list_buf: Handle<GlobalVariable>,
    out_buf: Handle<GlobalVariable>,

    config: CompilerConfig,

    pub(crate) assignments: HashMap<usize, Assignment>,
}

/// Index into the currently bound constant/input buffer. The value at this index at the time of invocation contains the indirect address of the value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedInputId(u32);

pub enum GpuInput {
    // Only supports scalars atm
    Varying(BaseType),
    Uniform(type_checker::Type, FixedInputId),
}
impl Compiler {
    fn new(
        config: CompilerConfig,
        global_assignments: TiVec<type_checker::AssignmentIndex, type_checker::Assignment>,
    ) -> Self {
        let mut module = Module::default();

        let ty_ctx = TyContext {
            u32: module.types.add_unspanned(naga::Type {
                name: "u32".to_string().into(),
                inner: naga::TypeInner::Scalar(naga::Scalar::U32),
            }),
            f32: module.types.add_unspanned(naga::Type {
                name: "f32".to_string().into(),
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            }),
            point: module.types.add_unspanned(naga::Type {
                name: "vec2".to_string().into(),
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar::F32,
                },
            }),
            bool: module.types.add_unspanned(naga::Type {
                name: "bool".to_string().into(),
                inner: naga::TypeInner::Scalar(naga::Scalar::BOOL),
            }),
            uvec3: module.types.add_unspanned(naga::Type {
                name: "uvec3".to_string().into(),
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar::U32,
                },
            }),
        };

        let arr_ty = module.types.add_unspanned(naga::Type {
            name: Some("Buffer".to_string()),
            inner: naga::TypeInner::Array {
                base: ty_ctx.u32,
                size: naga::ArraySize::Dynamic,
                stride: 4,
            },
        });
        let constant_buffer = module.global_variables.add_unspanned(GlobalVariable {
            name: "Constant Buffer".to_string().into(),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: arr_ty,
            init: None,
        });
        let list_buf = module.global_variables.add_unspanned(GlobalVariable {
            name: "Dynamic List Buffer".to_string().into(),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: arr_ty,
            init: None,
        });
        let out_buf = module.global_variables.add_unspanned(GlobalVariable {
            name: "Output Buffer".to_string().into(),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: 2,
            }),
            ty: arr_ty,
            init: None,
        });

        let global_assignments = global_assignments.into_iter().map(|v| (v.id, v)).collect();
        Self {
            module,
            types: ty_ctx,
            constant_buffer,
            list_buf,
            global_assignments,

            out_buf,
            assignments: HashMap::new(),
            config,
        }
    }

    fn compile_expr(&mut self, expr: &TypedExpression) {
        let mut func = CompilingFunction::new_primary(self);
        let ret = compile_scalar(self, &mut func, expr);

        let func = func.done(self, ret);
        self.module.entry_points.push(naga::EntryPoint {
            name: "capuronii_main".into(),
            stage: naga::ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [64, 1, 1],
            workgroup_size_overrides: None,
            function: func,
        });
    }
    fn scalar_type(&self, scalar: BaseType) -> Handle<naga::Type> {
        match scalar {
            BaseType::Number => self.types.f32,
            BaseType::Point => self.types.point,
            BaseType::Bool => self.types.bool,
            BaseType::Empty => panic!("invalid scalar type Empty"),
            BaseType::Polygon => todo!(),
        }
    }
    fn bind_list_assignment(&mut self, id: usize, list: MaterializedList) {
        self.assignments.insert(id, crate::Assignment::List(list));
    }
}

pub fn compile(expr: &TypedExpression) -> Module {
    let mut ctx = Compiler::new(Default::default(), ti_vec![]);
    ctx.compile_expr(expr);
    ctx.module
}
pub(crate) fn collect_list<'a>(expr: &'a TypedExpression) -> ListDef<'a> {
    let base = expr.ty.base();
    match &expr.e {
        type_checker::Expression::Identifier(_) => todo!(),
        type_checker::Expression::List(elements) => ListDef::new(
            base,
            listdef::UntypedListDef::LazyStatic(listdef::LazyStaticList { elements }),
        ),
        type_checker::Expression::ListRange {
            before_ellipsis,
            after_ellipsis,
        } => todo!(),
        type_checker::Expression::Broadcast {
            scalars,
            vectors,
            body,
        } => ListDef::new(
            base,
            listdef::UntypedListDef::Broadcast(LazyBroadcast {
                varying: vectors
                    .iter()
                    .map(|l| (l.id, collect_list(&l.value)))
                    .collect(),
                body,
                scalars,
            }),
        ),
        type_checker::Expression::BinaryOperation {
            operation,
            left,
            right,
        } => {
            let right = collect_list(right);
            let UntypedListDef::Broadcast(filter) = right.inner else {
                panic!()
            };
            match operation {
                type_checker::BinaryOperator::FilterNumberList
                | type_checker::BinaryOperator::FilterPointList => ListDef::new(
                    base,
                    listdef::UntypedListDef::Filter(Filter {
                        src: Box::new(collect_list(left)),
                        filter,
                    }),
                ),
                _ => unreachable!(),
            }
        }
        type_checker::Expression::Piecewise {
            test,
            consequent,
            alternate,
        } => ListDef::new(
            expr.ty.base(),
            UntypedListDef::Select(Select {
                test,
                consequent_alternate: Box::new((collect_list(consequent), collect_list(alternate))),
            }),
        ),
        type_checker::Expression::SumProd {
            kind,
            variable,
            lower_bound,
            upper_bound,
            body,
        } => todo!(),
        type_checker::Expression::For { body, lists } => ListDef::new(
            expr.ty.base(),
            UntypedListDef::Comprehension(LazyComprehension {
                varying: lists
                    .iter()
                    .map(|a| (a.id, collect_list(&a.value)))
                    .collect(),
                body,
            }),
        ),
        type_checker::Expression::BuiltIn { name, args } => match name {
            BuiltIn::JoinNumber | BuiltIn::JoinPoint => {
                let mut current_scalar_batch = 0..0;
                let mut listdefs = Vec::new();
                for (idx, val) in args.iter().enumerate() {
                    if val.ty.is_list() {
                        if !current_scalar_batch.is_empty() {
                            listdefs.push(ListDef::new(
                                val.ty.base(),
                                UntypedListDef::LazyStatic(LazyStaticList {
                                    elements: &args[current_scalar_batch.clone()],
                                }),
                            ));
                        }
                        current_scalar_batch.start = idx + 1;
                        current_scalar_batch.end = idx + 1;
                        listdefs.push(collect_list(val));
                    } else {
                        current_scalar_batch.end += 1;
                    }
                }
                if !current_scalar_batch.is_empty() {
                    listdefs.push(ListDef::new(
                        expr.ty.base(),
                        UntypedListDef::LazyStatic(LazyStaticList {
                            elements: &args[current_scalar_batch],
                        }),
                    ));
                }
                ListDef::new(
                    expr.ty.base(),
                    UntypedListDef::Join(Join { lists: listdefs }),
                )
            }

            _ => unimplemented!(),
        },
        _ => unreachable!(),
    }
}

fn compile_scalar(
    c: &mut Compiler,
    mut func: &mut CompilingFunction,
    expr: &TypedExpression,
) -> ScalarRef {
    let e = match &expr.e {
        type_checker::Expression::Number(v) => {
            naga::Expression::Literal(naga::Literal::F32(*v as f32))
        }
        type_checker::Expression::Identifier(i) => {
            if let Assignment::Scalar(s) = c.assignments.get(i).unwrap() {
                return ScalarRef::new(expr.ty.base(), func.load(*s));
            } else {
                panic!()
            }
        }
        type_checker::Expression::List(typed_expressions) => todo!(),
        type_checker::Expression::ListRange {
            before_ellipsis,
            after_ellipsis,
        } => todo!(),
        type_checker::Expression::Broadcast {
            scalars,
            vectors,
            body,
        } => todo!(),
        type_checker::Expression::UnaryOperation { operation, arg } => {
            let arg = compile_scalar(c, func, arg).inner;
            let un_math = |fun| naga::Expression::Math {
                fun,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            };
            let un_op = |op| naga::Expression::Unary { op, expr: arg };
            let swiz = |i: u8| naga::Expression::AccessIndex {
                base: arg,
                index: i as u32,
            };
            match operation {
                type_checker::UnaryOperator::NegNumber | type_checker::UnaryOperator::NegPoint => {
                    un_op(naga::UnaryOperator::Negate)
                }
                type_checker::UnaryOperator::Fac => todo!(),
                type_checker::UnaryOperator::Sqrt => un_math(MathFunction::Sqrt),
                type_checker::UnaryOperator::Abs => un_math(MathFunction::Abs),
                type_checker::UnaryOperator::Mag => un_math(MathFunction::Length),
                type_checker::UnaryOperator::PointX => swiz(0),
                type_checker::UnaryOperator::PointY => swiz(1),
            }
        }
        type_checker::Expression::BinaryOperation {
            operation,
            left,
            right,
        } => {
            if left.ty.is_list() {
                match operation {
                    type_checker::BinaryOperator::IndexPointList
                    | type_checker::BinaryOperator::IndexNumberList => {
                        func.push_frame(c);
                        let rhs = compile_scalar(c, func, right);
                        let lhs_list = collect_list(left);

                        let idx = func.make_index(rhs.inner);
                        let val = lhs_list.index(idx, c, &mut func);
                        func.pop_frame();
                        return WithScalarType::new(left.ty.base(), val);
                    }
                    _ => unreachable!(),
                }
            } else {
                let (left, right) = (
                    compile_scalar(c, func, left).inner,
                    compile_scalar(c, func, right).inner,
                );
                let bin_op = |op| naga::Expression::Binary { op, left, right };
                let bin_math = |fun| naga::Expression::Math {
                    fun,
                    arg: left,
                    arg1: Some(right),
                    arg2: None,
                    arg3: None,
                };
                match operation {
                    type_checker::BinaryOperator::AddPoint
                    | type_checker::BinaryOperator::AddNumber => bin_op(naga::BinaryOperator::Add),
                    type_checker::BinaryOperator::SubNumber
                    | type_checker::BinaryOperator::SubPoint => {
                        bin_op(naga::BinaryOperator::Subtract)
                    }
                    type_checker::BinaryOperator::MulNumber
                    | type_checker::BinaryOperator::MulPointNumber
                    | type_checker::BinaryOperator::MulNumberPoint => {
                        bin_op(naga::BinaryOperator::Multiply)
                    }
                    type_checker::BinaryOperator::DivPointNumber
                    | type_checker::BinaryOperator::DivNumber => {
                        bin_op(naga::BinaryOperator::Divide)
                    }

                    type_checker::BinaryOperator::Pow => bin_math(MathFunction::Pow),
                    type_checker::BinaryOperator::Dot => bin_math(MathFunction::Dot),
                    type_checker::BinaryOperator::Point => naga::Expression::Compose {
                        ty: c.types.point,
                        components: vec![left, right],
                    },
                    _ => unreachable!(),
                }
            }
        }
        type_checker::Expression::ChainedComparison {
            operands,
            operators,
        } => {
            let mut initial = compile_scalar(c, func, &operands[0]).inner;
            let mut expr = None;
            for (operand, operator) in operands[1..].iter().zip(operators) {
                let current = compile_scalar(c, func, operand).inner;

                let cond = naga::Expression::Binary {
                    op: match operator {
                        type_checker::ComparisonOperator::Equal => naga::BinaryOperator::Equal,
                        type_checker::ComparisonOperator::Less => naga::BinaryOperator::Less,
                        type_checker::ComparisonOperator::LessEqual => {
                            naga::BinaryOperator::LessEqual
                        }
                        type_checker::ComparisonOperator::Greater => naga::BinaryOperator::Greater,
                        type_checker::ComparisonOperator::GreaterEqual => {
                            naga::BinaryOperator::GreaterEqual
                        }
                    },
                    left: initial,
                    right: current,
                };
                initial = current;
                if let Some(to_or) = expr {
                    let to_or = func.add_unspanned(to_or);
                    let cond = func.add_unspanned(cond);
                    expr = Some(naga::Expression::Binary {
                        op: naga::BinaryOperator::LogicalOr,
                        left: to_or,
                        right: cond,
                    });
                } else {
                    expr = Some(cond);
                }
            }
            expr.expect("invalid comparison")
        }
        type_checker::Expression::Piecewise {
            test,
            consequent,
            alternate,
        } => {
            let [condition, accept, reject] =
                [test, consequent, alternate].map(|a| compile_scalar(c, func, a).inner);
            naga::Expression::Select {
                condition,
                accept,
                reject,
            }
        }
        type_checker::Expression::SumProd {
            kind,
            variable,
            lower_bound,
            upper_bound,
            body,
        } => todo!(),
        type_checker::Expression::BuiltIn { name, args } => match name {
            BuiltIn::Ln => todo!(),
            BuiltIn::Exp => todo!(),
            BuiltIn::Erf => todo!(),
            BuiltIn::Sin => todo!(),
            BuiltIn::Cos => todo!(),
            BuiltIn::Tan => todo!(),
            BuiltIn::Sec => todo!(),
            BuiltIn::Csc => todo!(),
            BuiltIn::Cot => todo!(),
            BuiltIn::Sinh => todo!(),
            BuiltIn::Cosh => todo!(),
            BuiltIn::Tanh => todo!(),
            BuiltIn::Sech => todo!(),
            BuiltIn::Csch => todo!(),
            BuiltIn::Coth => todo!(),
            BuiltIn::Asin => todo!(),
            BuiltIn::Acos => todo!(),
            BuiltIn::Atan => todo!(),
            BuiltIn::Atan2 => todo!(),
            BuiltIn::Asec => todo!(),
            BuiltIn::Acsc => todo!(),
            BuiltIn::Acot => todo!(),
            BuiltIn::Asinh => todo!(),
            BuiltIn::Acosh => todo!(),
            BuiltIn::Atanh => todo!(),
            BuiltIn::Asech => todo!(),
            BuiltIn::Acsch => todo!(),
            BuiltIn::Acoth => todo!(),
            BuiltIn::Abs => todo!(),
            BuiltIn::Sgn => todo!(),
            BuiltIn::Round => todo!(),
            BuiltIn::Floor => todo!(),
            BuiltIn::Ceil => todo!(),
            BuiltIn::Mod => todo!(),
            BuiltIn::Midpoint => todo!(),
            BuiltIn::Distance => todo!(),
            BuiltIn::Min => todo!(),
            BuiltIn::Max => todo!(),
            BuiltIn::Median => todo!(),
            BuiltIn::TotalNumber => todo!(),
            BuiltIn::TotalPoint => todo!(),
            BuiltIn::MeanNumber => todo!(),
            BuiltIn::MeanPoint => todo!(),
            BuiltIn::CountNumber | BuiltIn::CountPoint | BuiltIn::CountPolygon => {
                let l = collect_list(expr);
                let l = l.compute_len(c, func);
                return WithScalarType::new(BaseType::Number, func.u32_to_float(l));
            }
            BuiltIn::UniqueNumber => todo!(),
            BuiltIn::UniquePoint => todo!(),
            BuiltIn::UniquePolygon => todo!(),
            BuiltIn::Sort => todo!(),
            BuiltIn::SortKeyNumber => todo!(),
            BuiltIn::SortKeyPoint => todo!(),
            BuiltIn::SortKeyPolygon => todo!(),
            BuiltIn::Polygon => todo!(),
            BuiltIn::JoinNumber => todo!(),
            BuiltIn::JoinPoint => todo!(),
            BuiltIn::JoinPolygon => todo!(),
        },
        _ => unreachable!(),
    };
    let h = if e.needs_pre_emit() {
        func.add_preemit(e)
    } else {
        func.add_unspanned(e)
    };
    ScalarRef::new(expr.ty.base(), h)
}

#[derive(Debug, Clone, Copy)]
pub struct WithScalarType<T> {
    ty: BaseType,
    inner: T,
}
impl<T> WithScalarType<T> {
    pub fn new(ty: BaseType, inner: T) -> Self {
        Self { ty, inner }
    }
    pub fn types_eq<U>(&self, other: &WithScalarType<U>) -> bool {
        self.ty == other.ty
    }
}

pub(crate) type ScalarRef = WithScalarType<Handle<naga::Expression>>;

enum Assignment {
    Scalar(Handle<naga::Expression>),
    List(MaterializedList),
}

trait ArenaExt<T> {
    fn add_unspanned(&mut self, val: T) -> Handle<T>;
}
impl<T: Eq + std::hash::Hash> ArenaExt<T> for UniqueArena<T> {
    fn add_unspanned(&mut self, val: T) -> Handle<T> {
        self.insert(val, naga::Span::UNDEFINED)
    }
}
impl<T> ArenaExt<T> for Arena<T> {
    fn add_unspanned(&mut self, val: T) -> Handle<T> {
        self.append(val, naga::Span::UNDEFINED)
    }
}
