use std::{
    array,
    mem::{swap, take},
    panic::Location,
};

use log::trace;
use naga::{
    Block, Expression, Function, FunctionArgument, Handle, LocalVariable, Span, Statement, Type,
};
use parse::type_checker::{self, BaseType, TypedExpression};

use crate::{
    ArenaExt, Compiler, POINT_SIZE, ScalarValue,
    alloc::{Allocation, StaticAlloc},
    array_zip, collect_list, compile_scalar,
    listdef::ListDef,
    math_impl::Float32,
};

macro_rules! declare_consts {
    (
        $( #[$meta:meta] )*
        $vis:vis struct $struct_name:ident {
            $(
                $name:ident = $val:expr
            ),+ $(,)?
        }
    ) => {
        $( #[ $meta ] )*
        $vis struct $struct_name {
            $(
                pub(crate) $name: ::naga::Handle<::naga::Expression>
            ),+
        }
        impl $struct_name {
            $vis fn new(func: &mut ::naga::Function) -> Self {
                Self {
                    $(
                        $name: func.expressions.append(::naga::Expression::Literal(<_ as $crate::function::MakeConst>::make($val)), ::naga::Span::UNDEFINED)
                    ),+
                }
            }
        }
    };
}
trait MakeConst: Sized {
    fn make(self) -> naga::Literal;
}
impl MakeConst for f32 {
    fn make(self) -> naga::Literal {
        naga::Literal::F32(self)
    }
}
impl MakeConst for u32 {
    fn make(self) -> naga::Literal {
        naga::Literal::U32(self)
    }
}
impl MakeConst for bool {
    fn make(self) -> naga::Literal {
        naga::Literal::Bool(self)
    }
}
declare_consts! {
    #[allow(unused)]
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Constants {
        zero_f32 = 0.0,
        one_f32 = 1.0,
        max_f32 = f32::MAX,
        // log2(f32::MAX)
        log2_max_f32 = 128.0,
        // ln(f32::MAX)
        ln_max_f32 = 88.72284,
        canonical_nan = 0b111111111 << 22,
        zero_u32 = 0,
        one_u32 = 1,
        true_bool = true,
        false_bool = false,
        point_size_u32 = POINT_SIZE,
    }
}

#[derive(Debug)]
pub struct CompilingFunction {
    pub(crate) consts: Constants,
    pub(crate) invocation_id_u32: Handle<Expression>,
    pub(crate) frame_size_tallies: Vec<Option<Handle<Expression>>>,

    last_emit: usize,
    pub(crate) func: Function,
    // pointer to stack head
    pub(crate) stack_head: Handle<Expression>,
    // cache the current stack head when it hasn't been modified to avoid redundant loads
    stack_head_cache: Option<Handle<Expression>>,
    // pointer to stack buffer (ptr<device, [u32]>)
    pub(crate) stack_array_ptr: Handle<Expression>,
}

impl CompilingFunction {
    fn new(
        mut func: Function,
        stack_head: Handle<Expression>,
        stack_ptr: Handle<Expression>,
    ) -> Self {
        let constants = Constants::new(&mut func);
        let invocation_id_u32 = constants.zero_u32;
        Self {
            consts: constants,
            last_emit: 0,
            func,
            stack_head,
            stack_array_ptr: stack_ptr,
            stack_head_cache: None,
            invocation_id_u32,
            frame_size_tallies: Vec::new(),
        }
    }
    pub(crate) fn new_primary(ctx: &mut Compiler) -> Self {
        let mut func = Function {
            name: Some("capuronii_main".into()),
            arguments: vec![
                FunctionArgument {
                    name: "invocation_id".to_string().into(),
                    ty: ctx.types.uvec3,
                    binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
                },
                FunctionArgument {
                    name: "num_workgroups".to_string().into(),
                    ty: ctx.types.uvec3,
                    binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::NumWorkGroups)),
                },
            ],
            result: None,
            ..Default::default()
        };

        let inv_id = func
            .expressions
            .add_unspanned(Expression::FunctionArgument(0));
        let num_workgroups = func
            .expressions
            .add_unspanned(Expression::FunctionArgument(1));

        let mut gen_local = |name: &str, ty, expr| {
            let v = func.local_variables.add_unspanned(LocalVariable {
                name: name.to_string().into(),
                ty,
                init: expr,
            });
            func.expressions
                .add_unspanned(naga::Expression::LocalVariable(v))
        };
        let stack_base = gen_local("stack_base", ctx.types.u32, None);
        let stack_head = gen_local("stack_head", ctx.types.u32, None);

        let stack_ptr = func
            .expressions
            .add_unspanned(naga::Expression::GlobalVariable(ctx.list_buf));

        let mut func = Self::new(func, stack_head, stack_ptr);

        func.skip_emit_exprs();

        let (inv_idx, inv_idy, inv_idz) = (
            func.add_unspanned(naga::Expression::AccessIndex {
                base: inv_id,
                index: 0,
            }),
            func.add_unspanned(naga::Expression::AccessIndex {
                base: inv_id,
                index: 1,
            }),
            func.add_unspanned(naga::Expression::AccessIndex {
                base: inv_id,
                index: 2,
            }),
        );
        let (num_x, num_y) = (
            func.add_unspanned(naga::Expression::AccessIndex {
                base: num_workgroups,
                index: 0,
            }),
            func.add_unspanned(naga::Expression::AccessIndex {
                base: num_workgroups,
                index: 1,
            }),
        );
        let y = func.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: inv_idy,
            right: num_x,
        });
        let num_yx = func.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: num_x,
            right: num_y,
        });
        let z = func.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: inv_idz,
            right: num_yx,
        });
        let y_z = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: y,
            right: z,
        });
        let flattened_inv_id = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: inv_idx,
            right: y_z,
        });
        func.invocation_id_u32 = flattened_inv_id;
        let heap_per_invocation = func.add_preemit(naga::Expression::Literal(naga::Literal::U32(
            ctx.config.heap_per_invocation,
        )));
        let heap_offset = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: flattened_inv_id,
            right: heap_per_invocation,
        });
        func.store(stack_base, heap_offset);
        func.store(stack_head, heap_offset);
        func.push_frame();
        func
    }
    // gets the "serialized" representation (capable of representing all possible values) for this value
    pub(crate) fn serialize_scalar(
        &mut self,
        ctx: &Compiler,
        val: ScalarValue,
    ) -> Handle<Expression> {
        match val {
            ScalarValue::Number(float32) => float32.bits(self, ctx),
            ScalarValue::Point(arr) => {
                let components = arr.into_iter().map(|a| a.bits(self, ctx)).collect();
                self.add_unspanned(Expression::Compose {
                    ty: ctx.types.point_repr,
                    components,
                })
            }
            ScalarValue::Bool(handle) => handle,
        }
    }
    pub(crate) fn deserialize_scalar(
        &mut self,
        ctx: &Compiler,
        handle: Handle<Expression>,
        ty: BaseType,
    ) -> ScalarValue {
        match ty {
            BaseType::Number => ScalarValue::Number(Float32::from_bits(self, ctx, handle)),
            BaseType::Point => ScalarValue::Point(array::from_fn(|i| {
                let bits = self.access_fixed(handle, i.try_into().unwrap());
                Float32::from_bits(self, ctx, bits)
            })),
            BaseType::Bool => ScalarValue::Bool(handle),
            BaseType::Polygon => todo!(),
            BaseType::Empty => todo!(),
        }
    }
    fn flush_cache(&mut self) {
        self.stack_head_cache = None;
    }
    /// Switches the target for linear compilation to a new block, returning the old one.
    pub(crate) fn new_block(&mut self) -> Block {
        // barrier to make sure all expressions that were inserted in the old block get emitted within it
        self.emit_exprs();
        self.flush_cache();
        take(&mut self.func.body)
    }
    pub(crate) fn swap_block(&mut self, other: &mut Block) {
        self.emit_exprs();
        self.flush_cache();
        swap(&mut self.func.body, other);
    }
    #[track_caller]
    pub(crate) fn add_unspanned(&mut self, expr: naga::Expression) -> Handle<naga::Expression> {
        let ret = self.func.expressions.add_unspanned(expr);
        trace!("added expression {:#?}", ret);
        ret
    }
    #[track_caller]
    pub(crate) fn add_preemit(&mut self, expr: Expression) -> Handle<Expression> {
        self.emit_exprs();
        let r = self.add_unspanned(expr);
        self.skip_emit_exprs();
        r
    }
    pub(crate) fn load(&mut self, pointer: Handle<Expression>) -> Handle<Expression> {
        self.func
            .expressions
            .add_unspanned(Expression::Load { pointer })
    }
    #[track_caller]
    pub(crate) fn store(&mut self, pointer: Handle<Expression>, value: Handle<Expression>) {
        self.emit_exprs();
        self.func
            .body
            .push(Statement::Store { pointer, value }, Span::UNDEFINED);
    }

    pub(crate) fn bitcast_to_u32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::As {
            expr,
            kind: naga::ScalarKind::Uint,
            convert: None,
        })
    }
    pub(crate) fn bitcast_to_float(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::As {
            expr,
            kind: naga::ScalarKind::Float,
            convert: None,
        })
    }

    pub(crate) fn load_stack_head(&mut self) -> Handle<Expression> {
        match self.stack_head_cache {
            Some(v) => v,
            None => {
                let v = self.load(self.stack_head);
                self.stack_head_cache = Some(v);
                v
            }
        }
    }
    pub(crate) fn store_stack_head(&mut self, new_value: Handle<Expression>) {
        self.stack_head_cache = None;
        self.store(self.stack_head, new_value);
    }
    pub(crate) fn skip_emit_exprs(&mut self) {
        self.last_emit = self.func.expressions.len();
    }
    #[track_caller]
    pub(crate) fn emit_exprs(&mut self) {
        if self.last_emit == self.func.expressions.len() {
            return;
        }
        let range = self.func.expressions.range_from(self.last_emit);

        trace!("emit({:#?}) called", range.clone());

        self.func.body.push(Statement::Emit(range), Span::UNDEFINED);
        self.last_emit = self.func.expressions.len();
    }
    pub(crate) fn access_fixed(
        &mut self,
        val: Handle<Expression>,
        index: u32,
    ) -> Handle<Expression> {
        self.add_unspanned(Expression::AccessIndex { base: val, index })
    }
    pub(crate) fn increment(&mut self, val: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: val,
            right: self.consts.one_u32,
        })
    }
    pub(crate) fn size_of_scalar(&self, ty: BaseType) -> Handle<Expression> {
        match ty {
            BaseType::Number => self.consts.one_u32,
            BaseType::Point => self.consts.point_size_u32,
            BaseType::Bool => todo!(),
            BaseType::Empty => todo!(),
            BaseType::Polygon => todo!(),
        }
    }
    pub(crate) fn comptime_size_of_scalar(&self, ty: BaseType) -> u32 {
        match ty {
            BaseType::Number => 1,
            BaseType::Point => POINT_SIZE,
            BaseType::Polygon => todo!(),
            BaseType::Bool => todo!(),
            BaseType::Empty => todo!(),
        }
    }
    #[track_caller]
    pub(crate) fn new_local(
        &mut self,
        ty: Handle<Type>,
        init: Option<Handle<Expression>>,
        name: Option<String>,
    ) -> Handle<Expression> {
        let local_handle =
            self.func
                .local_variables
                .add_unspanned(LocalVariable { name, ty, init });
        trace!("created local {:#?}", local_handle);
        self.add_preemit(Expression::LocalVariable(local_handle))
    }
    pub(crate) fn new_local_index(&mut self, ctx: &Compiler) -> Handle<Expression> {
        self.new_local(
            ctx.types.u32,
            Some(self.consts.zero_u32),
            Some(format!("iteration_var_{}", self.func.local_variables.len())),
        )
    }
    pub(crate) fn get_scalar_assignment(
        &mut self,
        ctx: &mut Compiler,
        id: usize,
        scalar: BaseType,
    ) -> Handle<naga::Expression> {
        trace!("assignment: {id}, ty: {scalar:#?}");
        let s = ctx.scalar_type_repr(scalar);
        if let crate::Assignment::Scalar(s) = ctx.assignments.entry(id).or_insert_with(|| {
            crate::Assignment::Scalar(self.new_local(
                s,
                None,
                Some(format!(
                    "scalar_assignment_{}",
                    self.func.local_variables.len()
                )),
            ))
        }) {
            *s
        } else {
            panic!()
        }
    }
    pub(crate) fn bind_assignments(
        &mut self,
        ctx: &mut Compiler,
        assignments: &[type_checker::Assignment],
    ) {
        for assignment in assignments {
            if assignment.value.ty.is_list() {
                let def = collect_list(&assignment.value);
                let mat = def.materialize(ctx, self);
                ctx.bind_list_assignment(assignment.id, mat);
            } else {
                let assignment_addr =
                    self.get_scalar_assignment(ctx, assignment.id, assignment.value.ty.base());

                self.push_frame();
                let value = compile_scalar(ctx, self, &assignment.value);
                self.pop_frame();

                let repr = self.serialize_scalar(ctx, value);
                self.store(assignment_addr, repr);
            }
        }
    }
    pub(crate) fn select_scalar(
        &mut self,
        ctx: &Compiler,
        test: Handle<naga::Expression>,
        accept: ScalarValue,
        reject: ScalarValue,
    ) -> ScalarValue {
        match (accept, reject) {
            (ScalarValue::Number(accept), ScalarValue::Number(reject)) => {
                ScalarValue::Number(self.select(ctx, test, accept, reject))
            }
            (ScalarValue::Point(accept), ScalarValue::Point(reject)) => {
                ScalarValue::Point(array_zip([accept, reject], |[at, rt]| {
                    self.select(ctx, test, at, rt)
                }))
            }
            _ => unreachable!(),
        }
    }
    pub(crate) fn done(mut self, ctx: &mut Compiler, ret: ScalarValue) -> Function {
        self.pop_frame();
        let s = self.size_of_scalar(ret.ty());
        let addr = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: s,
            right: self.invocation_id_u32,
        });
        let out_ptr = self.add_preemit(Expression::GlobalVariable(ctx.out_buf));
        let out_alloc = StaticAlloc {
            array_pointer: out_ptr,
            size: 0,
        };
        self.store_to_alloc_typed(ctx, out_alloc, addr, ret);
        self.func
    }
    /// Indexes a ptr<array<T>> into a ptr<T>
    pub(crate) fn access(
        &mut self,
        ptr: Handle<Expression>,
        offset: Handle<Expression>,
    ) -> Handle<Expression> {
        self.add_unspanned(Expression::Access {
            base: ptr,
            index: offset,
        })
    }
    /// computes an index's corresponding raw index (some scalars may occupy multiple slots each)
    pub(crate) fn index_to_raw(
        &mut self,
        idx: Handle<Expression>,
        ty: BaseType,
    ) -> Handle<Expression> {
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: idx,
            right: self.size_of_scalar(ty),
        })
    }
    pub(crate) fn nan_value_for_ty(&mut self, ctx: &Compiler, val: BaseType) -> ScalarValue {
        let nan = self.new_literal(ctx, f32::NAN);
        match val {
            BaseType::Number => nan,
            BaseType::Point => ScalarValue::Point([nan.num(); 2]),
            BaseType::Polygon => todo!(),
            BaseType::Bool => todo!(),
            BaseType::Empty => todo!(),
        }
    }
    pub(crate) fn len_from_size(
        &mut self,
        alloc: impl Allocation,
        ty: BaseType,
    ) -> Handle<Expression> {
        match alloc.size() {
            crate::alloc::AllocSize::Dynamic(handle) => match ty {
                BaseType::Number => handle,
                BaseType::Point => self.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Divide,
                    left: handle,
                    right: self.consts.point_size_u32,
                }),
                BaseType::Polygon => todo!(),
                BaseType::Bool => todo!(),
                BaseType::Empty => todo!(),
            },
            crate::alloc::AllocSize::Static(val) => match ty {
                BaseType::Number => self.add_preemit(Expression::Literal(naga::Literal::U32(val))),
                BaseType::Point => {
                    self.add_preemit(Expression::Literal(naga::Literal::U32(val / POINT_SIZE)))
                }
                BaseType::Polygon => todo!(),
                BaseType::Bool => todo!(),
                BaseType::Empty => todo!(),
            },
        }
    }

    pub(crate) fn store_to_top_of_stack_typed(&mut self, ctx: &Compiler, value: ScalarValue) {
        let s = self.size_of_scalar(value.ty());
        // "allocate" a single item to advance the stack pointer
        let single_alloc = self.alloc_stack(ctx, s);
        // and store
        self.store_to_alloc_typed(ctx, single_alloc, self.consts.zero_u32, value);
    }
    pub(crate) fn store_to_alloc_typed(
        &mut self,
        ctx: &Compiler,
        alloc: impl Allocation,
        mut offset: Handle<Expression>,
        val: ScalarValue,
    ) {
        match val {
            ScalarValue::Number(float32) => {
                let bits = float32.bits(self, ctx);
                alloc.store_to(self, offset, bits);
            }
            ScalarValue::Point(coordinates) => {
                for coord in coordinates.into_iter() {
                    let bits = coord.bits(self, ctx);
                    alloc.store_to(self, offset, bits);
                    offset = self.increment(offset);
                }
            }
            ScalarValue::Bool(_) => panic!("list of bool is not supported"),
        }
    }
    pub(crate) fn load_from_alloc_typed(
        &mut self,
        ctx: &Compiler,
        alloc: impl Allocation,
        mut offset: Handle<Expression>,
        ty: BaseType,
    ) -> ScalarValue {
        match ty {
            BaseType::Number => {
                let bits = alloc.load_at(self, offset);
                ScalarValue::Number(Float32::from_bits(self, ctx, bits))
            }
            BaseType::Point => ScalarValue::Point(array::from_fn(|_| {
                let s = self.size_of_scalar(ty);
                let bits = alloc.load_at(self, offset);
                // advance pointer
                offset = self.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: offset,
                    right: s,
                });
                Float32::from_bits(self, ctx, bits)
            })),
            BaseType::Polygon => todo!(),
            BaseType::Bool => panic!("list of bool is not supported"),
            BaseType::Empty => todo!(),
        }
    }
    pub(crate) fn zeroed(&mut self, ty: BaseType) -> ScalarValue {
        let z = Float32::new_assume_finite(self.consts.zero_f32);
        match ty {
            BaseType::Number => ScalarValue::Number(z),
            BaseType::Point => ScalarValue::Point(array::from_fn(|_| z)),
            BaseType::Polygon => todo!(),
            BaseType::Bool => ScalarValue::Bool(self.consts.false_bool),
            BaseType::Empty => todo!(),
        }
    }
    pub(crate) fn get_boxed_bits(
        &mut self,
        ctx: &Compiler,
        val: ScalarValue,
    ) -> Box<[Handle<Expression>]> {
        match val {
            ScalarValue::Number(float32) => Box::new([float32.bits(self, ctx)]),
            ScalarValue::Point(a) => a.map(|v| v.bits(self, ctx)).into(),
            ScalarValue::Bool(_) => todo!(),
        }
    }
    pub(crate) fn u32_to_float(&mut self, number: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::As {
            expr: number,
            kind: naga::ScalarKind::Float,
            convert: Some(4),
        })
    }
    pub(crate) fn push_statement(&mut self, s: Statement) {
        self.func.body.push(s, Span::UNDEFINED);
    }
}
