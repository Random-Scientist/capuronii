use naga::{
    AddressSpace, Arena, GlobalVariable, Handle, Module, ResourceBinding, StorageAccess,
    UniqueArena,
};
use parse::type_checker::{self, BaseType, BuiltIn, TypedExpression};
use std::{collections::HashMap, mem::MaybeUninit};
use typed_index_collections::{TiVec, ti_vec};

use crate::{
    function::CompilingFunction,
    listdef::{
        Filter, Join, LazyBroadcast, LazyComprehension, LazyStaticList, ListDef, MaterializedList,
        Select, UntypedListDef,
    },
    math_impl::{Float32, NumericConfig},
};

mod alloc;
mod function;
mod listdef;
mod math_impl;
mod symath;

#[cfg(test)]
mod test;

pub(crate) const POINT_SIZE: u32 = 2;
#[derive(Debug, Clone, Copy)]
pub enum ScalarValue {
    Number(Float32),
    Point([Float32; POINT_SIZE as usize]),
    Bool(Handle<naga::Expression>),
}
impl ScalarValue {
    fn ty(&self) -> BaseType {
        match self {
            ScalarValue::Number(_) => BaseType::Number,
            ScalarValue::Point(_) => BaseType::Point,
            ScalarValue::Bool(_) => BaseType::Bool,
        }
    }
    fn expect_number(self) -> Float32 {
        match self {
            ScalarValue::Number(n) => n,
            ScalarValue::Point(_) | ScalarValue::Bool(_) => {
                panic!("expected a number, got a point or bool")
            }
        }
    }
    fn expect_point(self) -> [Float32; POINT_SIZE as usize] {
        match self {
            ScalarValue::Number(_) | ScalarValue::Bool(_) => {
                panic!("expected a point, got a number or bool")
            }
            ScalarValue::Point(p) => p,
        }
    }
    fn expect_bool(self) -> Handle<naga::Expression> {
        match self {
            ScalarValue::Number(_) | ScalarValue::Point(_) => {
                panic!("expected a bool, got a number or point")
            }
            ScalarValue::Bool(h) => h,
        }
    }
}

pub struct CompilerConfig {
    /// Maximum size of the arena allocated for each invocation in intervals of 4 bytes (e.g. setting this value to 10 implies a >=40 byte long backing heap buffer)
    pub heap_per_invocation: u32,
    /// Map of [`Identifier`](parse::type_checker::Expression::Identifier)s to override with a dynamic GPU value
    pub io_map: Vec<(usize, GpuInput)>,
    /// Configurable behavior for numerical operations
    pub numeric: NumericConfig,
}
impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            heap_per_invocation: 20000,
            io_map: Vec::new(),
            numeric: Default::default(),
        }
    }
}

struct TyContext {
    u32: Handle<naga::Type>,
    f32: Handle<naga::Type>,
    point_repr: Handle<naga::Type>,
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
            point_repr: module.types.add_unspanned(naga::Type {
                name: "uvec2".to_string().into(),
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    // must be able to represent NaN
                    scalar: naga::Scalar::U32,
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
            BaseType::Point => self.types.point_repr,
            BaseType::Bool => self.types.bool,
            BaseType::Empty => panic!("invalid scalar type Empty"),
            BaseType::Polygon => todo!(),
        }
    }
    fn scalar_type_repr(&self, scalar: BaseType) -> Handle<naga::Type> {
        match scalar {
            BaseType::Number => self.types.u32,
            BaseType::Point => self.types.point_repr,
            BaseType::Polygon => todo!(),
            BaseType::Bool => todo!(),
            BaseType::Empty => todo!(),
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
        type_checker::Expression::ListRange { .. } => todo!(),
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
        type_checker::Expression::SumProd { .. } => todo!(),
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
    ctx: &mut Compiler,
    func: &mut CompilingFunction,
    expr: &TypedExpression,
) -> ScalarValue {
    match &expr.e {
        type_checker::Expression::Number(v) => func.new_literal(*v as f32),
        type_checker::Expression::Identifier(i) => {
            assert!(!expr.ty.is_list());
            let ty = expr.ty.base();
            let assignment = func.get_scalar_assignment(ctx, *i, ty);
            let ser = func.load(assignment);
            func.deserialize_scalar(ctx, ser, ty)
        }
        type_checker::Expression::UnaryOperation { operation, arg } => {
            let arg = compile_scalar(ctx, func, arg);
            'number: {
                return ScalarValue::Number(match operation {
                    type_checker::UnaryOperator::NegNumber => func.negate(ctx, arg.expect_number()),
                    type_checker::UnaryOperator::Fac => todo!(),
                    type_checker::UnaryOperator::Sqrt => func.sqrt(ctx, arg.expect_number()),
                    type_checker::UnaryOperator::Abs => func.abs(ctx, arg.expect_number()),
                    type_checker::UnaryOperator::PointX => arg.expect_point()[0],
                    type_checker::UnaryOperator::PointY => arg.expect_point()[1],
                    type_checker::UnaryOperator::Mag => arg
                        .expect_point()
                        .map(|coord| func.mul(ctx, coord, coord))
                        .into_iter()
                        .fold_on(|r, l| func.add(ctx, r, l))
                        .expect("point length may not be zero"),
                    _ => break 'number,
                });
            }
            ScalarValue::Point(match operation {
                type_checker::UnaryOperator::NegPoint => {
                    arg.expect_point().map(|coord| func.negate(ctx, coord))
                }
                _ => unreachable!(),
            })
        }
        type_checker::Expression::BinaryOperation {
            operation,
            left,
            right,
        } => {
            if left.ty.is_list() {
                match operation {
                    type_checker::BinaryOperator::IndexNumberList => todo!(),
                    type_checker::BinaryOperator::IndexPointList => todo!(),
                    type_checker::BinaryOperator::IndexPolygonList => todo!(),
                    type_checker::BinaryOperator::FilterNumberList => todo!(),
                    type_checker::BinaryOperator::FilterPointList => todo!(),
                    type_checker::BinaryOperator::FilterPolygonList => todo!(),
                    _ => unreachable!(),
                }
            } else {
                let (left, right) = (
                    compile_scalar(ctx, func, left),
                    compile_scalar(ctx, func, right),
                );
                ScalarValue::Point('point: {
                    return ScalarValue::Number(match operation {
                        type_checker::BinaryOperator::AddNumber => {
                            func.add(ctx, left.expect_number(), right.expect_number())
                        }
                        type_checker::BinaryOperator::AddPoint => {
                            break 'point array_zip(
                                [left.expect_point(), right.expect_point()],
                                |[lc, rc]| func.add(ctx, lc, rc),
                            );
                        }
                        type_checker::BinaryOperator::SubNumber => {
                            func.sub(ctx, left.expect_number(), right.expect_number())
                        }
                        type_checker::BinaryOperator::SubPoint => {
                            break 'point array_zip(
                                [left.expect_point(), right.expect_point()],
                                |[lc, rc]| func.sub(ctx, lc, rc),
                            );
                        }
                        type_checker::BinaryOperator::MulNumber => {
                            func.mul(ctx, left.expect_number(), right.expect_number())
                        }
                        type_checker::BinaryOperator::MulPointNumber => {
                            break 'point array_zip(
                                [
                                    left.expect_point(),
                                    std::array::from_fn(|_| right.expect_number()),
                                ],
                                |[lc, rc]| func.mul(ctx, lc, rc),
                            );
                        }
                        type_checker::BinaryOperator::MulNumberPoint => todo!(),
                        type_checker::BinaryOperator::DivNumber => {
                            func.div(ctx, left.expect_number(), right.expect_number())
                        }
                        type_checker::BinaryOperator::DivPointNumber => {
                            break 'point array_zip(
                                [
                                    left.expect_point(),
                                    std::array::from_fn(|_| right.expect_number()),
                                ],
                                |[lc, rc]| func.div(ctx, lc, rc),
                            );
                        }
                        type_checker::BinaryOperator::Pow => todo!(),
                        type_checker::BinaryOperator::Dot => todo!(),

                        type_checker::BinaryOperator::Point => {
                            break 'point [left.expect_number(), right.expect_number()];
                        }
                        _ => unreachable!(),
                    });
                })
            }
        }
        type_checker::Expression::ChainedComparison {
            operands,
            operators,
        } => {
            let mut initial = compile_scalar(ctx, func, &operands[0]).expect_number();
            let mut expr = None;
            for (operand, operator) in operands[1..].iter().zip(operators) {
                let current = compile_scalar(ctx, func, operand).expect_number();

                let cond = func.cmp(ctx, *operator, initial, current);
                initial = current;
                if let Some(to_or) = expr {
                    expr = Some(func.add_unspanned(naga::Expression::Binary {
                        op: naga::BinaryOperator::LogicalOr,
                        left: to_or,
                        right: cond,
                    }));
                } else {
                    expr = Some(cond);
                }
            }
            ScalarValue::Bool(expr.expect("invalid comparison"))
        }
        type_checker::Expression::Piecewise {
            test,
            consequent,
            alternate,
        } => {
            let [condition, accept, reject] =
                [test, consequent, alternate].map(|a| compile_scalar(ctx, func, a));
            let ty = accept.ty();
            assert!(ty == reject.ty(), "divergent piecewise return type");

            let accept = func.serialize_scalar(ctx, accept);
            let reject = func.serialize_scalar(ctx, reject);
            let s = func.add_unspanned(naga::Expression::Select {
                condition: condition.expect_bool(),
                accept,
                reject,
            });
            func.deserialize_scalar(ctx, s, ty)
        }
        type_checker::Expression::SumProd { .. } => todo!(),
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
                let l = collect_list(&args[0]);
                let len = l.compute_len(ctx, func);
                ScalarValue::Number(Float32::new_assume_finite(func.u32_to_float(len)))
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
        type_checker::Expression::List(_)
        | type_checker::Expression::ListRange { .. }
        | type_checker::Expression::For { .. }
        | type_checker::Expression::Broadcast { .. } => unreachable!(),
    }
}

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

impl<T, U: Iterator<Item = T>> IteratorExt<T> for U {}
trait IteratorExt<T>: Iterator<Item = T> + Sized {
    fn fold_on(mut self, func: impl FnMut(T, T) -> T) -> Option<T> {
        let s = self.next()?;
        Some(self.fold(s, func))
    }
}
/// Zip two const arrays of the same length. Likely explodes the stack for large array sizes, so don't do that
fn array_zip<const L: usize, const N: usize, T: Copy>(
    arrs: [[T; L]; N],
    mut f: impl FnMut([T; N]) -> T,
) -> [T; L] {
    let mut out = [const { MaybeUninit::<T>::uninit() }; L];
    for idx in 0..L {
        let val = f(arrs.map(|v| v[idx]));
        out[idx] = MaybeUninit::new(val);
    }
    // Safety: all fields are init after iteration
    unsafe { transmute_unchecked(out) }
}
pub(crate) const unsafe fn transmute_unchecked<Src, Dst>(value: Src) -> Dst {
    union Transmute<Src, Dst> {
        src: ::core::mem::ManuallyDrop<Src>,
        dst: ::core::mem::ManuallyDrop<Dst>,
    }
    // Safety: caller
    ::core::mem::ManuallyDrop::into_inner(unsafe {
        Transmute {
            src: ::core::mem::ManuallyDrop::new(value),
        }
        .dst
    })
}
