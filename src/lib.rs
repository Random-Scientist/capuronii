use std::{
    collections::HashMap,
    mem::{self, discriminant},
};

use naga::{
    AddressSpace, Arena, Block, Function, FunctionArgument, FunctionResult, GlobalVariable, Handle,
    LocalVariable, MathFunction, Module, ResourceBinding, Statement, StorageAccess, StructMember,
    UniqueArena,
};
use parse::type_checker::{self, BaseType, TypedExpression};
use typed_index_collections::{TiSlice, TiVec, ti_vec};

use crate::{
    alloc::{StackAlloc, StackState},
    function::CompilingFunction,
};
mod alloc;
mod function;
#[cfg(test)]
mod test;
struct TyContext {
    u32: Handle<naga::Type>,
    f32: Handle<naga::Type>,
    point: Handle<naga::Type>,
    bool: Handle<naga::Type>,
    uvec3: Handle<naga::Type>,
    stack_head_ptr: Handle<naga::Type>,
}

pub struct Iterator {
    state: Handle<LocalVariable>,
    next: Handle<Function>,
    get_size: Handle<Function>,
    ty: BaseType,
}

enum IterTypes {}
/// Spec Iterator<T: Scalar>
/// {
///     type State;
///     // amortized O(1)
///     index(state: &mut State, idx: u32) -> T;
///     // amortized O(1)
///     len(state: &mut State) -> u32;
///     
/// }

struct RangeList {
    state: Handle<LocalVariable>,
}

struct Compiler {
    global_assignments: HashMap<usize, type_checker::Assignment>,
    module: Module,
    ty_ctx: TyContext,
    uniforms: Handle<GlobalVariable>,
    list_buf: Handle<GlobalVariable>,
    heap_per_invocation: u32,
    stack: StackState,
    // each value in the map is of type ptr<function, Scalar> and points to a local unique to a given assignment binding
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
        heap_per_invocation: u32,
        global_assignments: TiVec<type_checker::AssignmentIndex, type_checker::Assignment>,
        io_map: HashMap<usize, GpuInput>,
    ) -> Self {
        let mut module = Module::default();
        let _u32 = module.types.add_unspanned(naga::Type {
            name: "Int".to_string().into(),
            inner: naga::TypeInner::Scalar(naga::Scalar::U32),
        });
        let ty_ctx = TyContext {
            u32: _u32,
            f32: module.types.add_unspanned(naga::Type {
                name: "Number".to_string().into(),
                inner: naga::TypeInner::Scalar(naga::Scalar::F32),
            }),
            point: module.types.add_unspanned(naga::Type {
                name: "Point".to_string().into(),
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar::F32,
                },
            }),
            bool: module.types.add_unspanned(naga::Type {
                name: "Bool".to_string().into(),
                inner: naga::TypeInner::Scalar(naga::Scalar::BOOL),
            }),
            uvec3: module.types.add_unspanned(naga::Type {
                name: "uvec3".to_string().into(),
                inner: naga::TypeInner::Vector {
                    size: naga::VectorSize::Tri,
                    scalar: naga::Scalar::U32,
                },
            }),
            stack_head_ptr: module.types.add_unspanned(naga::Type {
                name: Some(format!("StackHeadPtr")),
                inner: naga::TypeInner::Pointer {
                    base: _u32,
                    space: AddressSpace::Function,
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
        let uniforms = module.global_variables.add_unspanned(GlobalVariable {
            name: "Constant List Buffer".to_string().into(),
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

        let stack = StackState::new();
        let global_assignments = global_assignments.into_iter().map(|v| (v.id, v)).collect();
        Self {
            module,
            ty_ctx,
            uniforms,
            list_buf,
            stack,
            global_assignments,
            heap_per_invocation,
            assignments: HashMap::new(),
        }
    }

    fn compile_expr(&mut self, expr: &TypedExpression) -> Handle<Function> {
        let mut func = CompilingFunction::new_primary(self);
        let ret = compile_scalar(self, &mut func, expr);

        let mut func = func.done(self, ret);
        self.module.functions.add_unspanned(func)
    }
    fn scalar_type(&self, scalar: BaseType) -> Handle<naga::Type> {
        match scalar {
            BaseType::Number => self.ty_ctx.f32,
            BaseType::Point => self.ty_ctx.point,
            BaseType::Bool => self.ty_ctx.bool,
            BaseType::Empty => panic!("invalid scalar type Empty"),
        }
    }
}

pub fn compile(expr: &TypedExpression) -> Module {
    let mut ctx = Compiler::new(20000, ti_vec![], HashMap::new());
    ctx.compile_expr(expr);
    ctx.module
}
fn materialize_list(
    c: &mut Compiler,
    func: &mut CompilingFunction,
    expr: &TypedExpression,
) -> StackList {
    match &expr.e {
        type_checker::Expression::Identifier(i) => {
            let Assignment::List(a) = *c.assignments.get(i).unwrap() else {
                panic!()
            };
            StackList::new(expr.ty.base(), a)
        }
        type_checker::Expression::List(typed_expressions) => {
            let len = func.add_preemit(naga::Expression::Literal(naga::Literal::U32(
                typed_expressions.len() as u32,
            )));
            let list = func.alloc_list(c, expr.ty.base(), len);

            for (idx, s) in typed_expressions.iter().enumerate() {
                let value = compile_scalar(c, func, s);
                let index =
                    func.add_preemit(naga::Expression::Literal(naga::Literal::U32(idx as u32)));
                func.store_typed(&list, index, value);
            }
            list
        }
        type_checker::Expression::ListRange {
            before_ellipsis,
            after_ellipsis,
        } => todo!(),
        type_checker::Expression::Broadcast {
            scalars,
            vectors,
            body,
        } => {
            for scalar in scalars {
                let value = compile_scalar(c, func, &scalar.value).inner;
                let local = func.new_scalar_assignment(c, scalar.id, scalar.value.ty.base());
                func.store(local, value);
            }
            let lists = vectors
                .iter()
                .map(|a| {
                    let l = materialize_list(c, func, &a.value);

                    (func.new_scalar_assignment(c, a.id, l.ty), l)
                })
                .collect::<Vec<_>>();

            let mut len = None;

            for (id, l) in lists.iter() {
                let this_len = func.compute_list_len(l);
                if let Some(l) = len {
                    len = Some(func.add_unspanned(naga::Expression::Math {
                        fun: MathFunction::Min,
                        arg: l,
                        arg1: Some(this_len),
                        arg2: None,
                        arg3: None,
                    }));
                } else {
                    len = Some(this_len);
                }
            }
            // allocate output list
            let len = len.unwrap();
            let out = func.alloc_list(c, expr.ty.base(), len);

            let index = func.new_local(c.ty_ctx.u32, Some(func.zero_u32));

            // compile into a fresh block for the loop body
            let mut prev_body = func.new_block();

            // new stack frame for scalar eval
            func.push_frame(c);
            // loop body prologue
            // load iter index
            let iter_idx = func.load(index);

            for (assignment_pointer, list) in lists.iter() {
                // assign varyings
                let access = func.load_typed(c, list, iter_idx);
                func.store(*assignment_pointer, access.inner);
            }
            let res = compile_scalar(c, func, body);
            func.store_typed(&out, iter_idx, res);
            // end stack frame, we already stored out the result we care about
            func.pop_frame(c);

            let next_idx = func.increment(iter_idx);
            let should_break = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::GreaterEqual,
                left: next_idx,
                right: len,
            });
            // barrier
            func.emit_exprs();

            func.body.push(
                Statement::If {
                    condition: should_break,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                },
                naga::Span::UNDEFINED,
            );
            // increment index
            func.store(index, next_idx);

            func.swap_block(&mut prev_body);
            func.body.push(
                Statement::Loop {
                    body: prev_body,
                    continuing: Block::new(),
                    break_if: None,
                },
                naga::Span::UNDEFINED,
            );
            out
        }

        type_checker::Expression::ChainedComparison {
            operands,
            operators,
        } => todo!(),
        type_checker::Expression::Piecewise {
            test,
            consequent,
            alternate,
        } => {
            let test = compile_scalar(c, func, &test);

            let len = func.new_local(c.ty_ctx.u32, None);
            let base_addr = func.new_local(c.ty_ctx.u32, None);

            let old_body = func.new_block();

            let consequent_list = materialize_list(c, func, &consequent);
            func.store(base_addr, consequent_list.inner.base_addr);
            func.store(len, consequent_list.inner.len);

            let consequent_body = func.new_block();

            let alternate_list = materialize_list(c, func, &alternate);
            func.store(base_addr, alternate_list.inner.base_addr);
            func.store(len, alternate_list.inner.len);

            let mut alternate_body = old_body;
            func.swap_block(&mut alternate_body);
            func.body.push(
                Statement::If {
                    condition: test.inner,
                    accept: consequent_body,
                    reject: alternate_body,
                },
                naga::Span::UNDEFINED,
            );
            let base_addr = func.load(base_addr);
            let len = func.load(len);

            StackList::new(
                consequent.ty,
                StackAlloc {
                    stack_scope: todo!(),
                    base_addr,
                    len,
                },
            )
        }
        type_checker::Expression::SumProd {
            kind,
            variable,
            lower_bound,
            upper_bound,
            body,
        } => todo!(),
        type_checker::Expression::BinaryOperation {
            operation:
                type_checker::BinaryOperator::FilterNumberList
                | type_checker::BinaryOperator::FilterPointList,
            left,
            right,
        } => {
            todo!()
        }
        type_checker::Expression::For { body, lists } => {
            let lists = lists
                .iter()
                .map(|a| {
                    let l = materialize_list(c, func, &a.value);

                    (func.new_scalar_assignment(c, a.id, l.ty), l)
                })
                .collect::<Vec<_>>();
            let mut out_len = None;
            let mut lengths = Vec::new();

            for (id, l) in lists.iter() {
                let this_len = func.compute_list_len(l);
                lengths.push(this_len);
                if let Some(l) = out_len {
                    out_len = Some(func.add_unspanned(naga::Expression::Binary {
                        op: naga::BinaryOperator::Multiply,
                        left: l,
                        right: this_len,
                    }));
                } else {
                    out_len = Some(this_len);
                }
            }
            // allocate output list
            let len = out_len.unwrap();
            let out = func.alloc_list(c, expr.ty.base(), len);

            let index = func.new_local(c.ty_ctx.u32, Some(func.zero_u32));

            // compile into a new block
            let mut prev_body = func.new_block();

            // new stack frame for comprehension body
            func.push_frame(c);

            // bind varyings
            let mut prev_section_len = func.one_u32;
            let iter_index = func.load(index);
            for ((local, value), length) in lists.iter().zip(lengths) {
                // integer division truncates towards zero, which is equivalent to `div_floor` (the desired primitive here) for whole number values
                let div = func.add_unspanned(naga::Expression::Binary {
                    op: naga::BinaryOperator::Divide,
                    left: iter_index,
                    right: prev_section_len,
                });
                let index = func.add_unspanned(naga::Expression::Binary {
                    op: naga::BinaryOperator::Modulo,
                    left: div,
                    right: length,
                });
                prev_section_len = func.add_preemit(naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    left: prev_section_len,
                    right: length,
                });
                let val = func.load_typed(c, value, index);
                func.store(*local, val.inner);
            }

            // bind assignments
            for assignment in body.assignments.iter() {
                if assignment.value.ty.is_list() {
                    let list = materialize_list(c, func, &assignment.value);
                    func.new_list_assignment(c, assignment.id, &list.inner);
                } else {
                    let p =
                        func.new_scalar_assignment(c, assignment.id, assignment.value.ty.base());
                    //new frame for scalar eval
                    func.push_frame(c);
                    let val = compile_scalar(c, func, &assignment.value);
                    // assign out
                    func.store(p, val.inner);
                    // discard stack scope, we read our scalar out so have no use for it
                    func.pop_frame(c);
                }
            }
            // eval body
            let scalar = compile_scalar(c, func, &body.value);

            func.store_typed(&out, iter_index, scalar);

            // pop scope
            func.pop_frame(c);

            let next_idx = func.increment(iter_index);

            let should_break = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::GreaterEqual,
                left: next_idx,
                right: len,
            });

            // barrier
            func.emit_exprs();

            func.body.push(
                Statement::If {
                    condition: should_break,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                },
                naga::Span::UNDEFINED,
            );
            // increment index
            func.store(index, next_idx);
            func.swap_block(&mut prev_body);
            func.body.push(
                Statement::Loop {
                    body: prev_body,
                    continuing: Block::new(),
                    break_if: None,
                },
                naga::Span::UNDEFINED,
            );
            out
        }

        type_checker::Expression::BuiltIn(built_in) => todo!(),
        _ => unreachable!(),
    }
}
// fn compile_list<'e>(
//     c: &mut Compiler,
//     func: &mut CompilingFunction,
//     expr: &'e TypedExpression,
// ) -> ListDef<'e> {
//     match &expr.e {
//         type_checker::Expression::Identifier(_) => todo!(),
//         type_checker::Expression::List(typed_expressions) => ListDef::Scalars(typed_expressions),
//         type_checker::Expression::ListRange {
//             before_ellipsis,
//             after_ellipsis,
//         } => todo!(),
//         type_checker::Expression::Broadcast {
//             scalars,
//             vectors,
//             body,
//         } => ListDef::Broadcast(Broadcast {
//             body,
//             over: vectors
//                 .iter()
//                 .map(|a| (a.id, compile_list(c, func, &a.value)))
//                 .collect(),
//             scalars,
//         }),

//         type_checker::Expression::ChainedComparison {
//             operands,
//             operators,
//         } => todo!(),
//         type_checker::Expression::Piecewise {
//             test,
//             consequent,
//             alternate,
//         } => ListDef::Piecewise(Piecewise {
//             test,
//             values: [consequent, alternate]
//                 .map(|expr| compile_list(c, func, expr))
//                 .into(),
//         }),
//         type_checker::Expression::SumProd {
//             kind,
//             variable,
//             lower_bound,
//             upper_bound,
//             body,
//         } => todo!(),
//         type_checker::Expression::BinaryOperation {
//             operation:
//                 type_checker::BinaryOperator::FilterNumberList
//                 | type_checker::BinaryOperator::FilterPointList,
//             left,
//             right,
//         } => {
//             let ListDef::Broadcast(filter) = compile_list(c, func, right) else {
//                 panic!()
//             };
//             ListDef::Filter(Filter {
//                 filter,
//                 source: Box::new(compile_list(c, func, left)),
//             })
//         }
//         type_checker::Expression::For { body, lists } => todo!(),

//         type_checker::Expression::BuiltIn(built_in) => todo!(),
//         _ => unreachable!(),
//     }
// }
fn compile_scalar(
    c: &mut Compiler,
    func: &mut CompilingFunction,
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
                        let rhs = compile_scalar(c, func, right);
                        let lhs = materialize_list(c, func, left);

                        let idx = func.make_index(rhs.inner);
                        return func.load_typed(c, &lhs, idx);
                    }
                    type_checker::BinaryOperator::FilterNumberList => todo!(),
                    type_checker::BinaryOperator::FilterPointList => todo!(),
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
                        ty: c.ty_ctx.point,
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
                [test, consequent, alternate].map(|a| compile_scalar(c, func, &a).inner);
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
        type_checker::Expression::For { body, lists } => todo!(),
        type_checker::Expression::BuiltIn(built_in) => todo!(),
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
pub(crate) type StackList = WithScalarType<StackAlloc>;
enum Assignment {
    Scalar(Handle<naga::Expression>),
    List(StackAlloc),
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
