use std::{
    array,
    mem::{swap, take},
    ops::{Deref, DerefMut},
    panic::Location,
};

use naga::{
    Block, Expression, Function, FunctionArgument, Handle, LocalVariable, Span, Statement, Type,
};
use parse::type_checker::{self, BaseType};

use crate::{
    ArenaExt, Compiler, ScalarRef, StackList, alloc::StackAlloc, collect_list, compile_scalar,
};

const POINT_SIZE: u32 = 2;
#[derive(Debug, Clone, Copy)]
pub(crate) struct Constants {
    pub(crate) zero_f32: Handle<Expression>,
    pub(crate) one_f32: Handle<Expression>,
    pub(crate) max_f32: Handle<Expression>,
    pub(crate) log2_max_f32: Handle<Expression>,
    pub(crate) canonical_nan: Handle<Expression>,
    pub(crate) zero_u32: Handle<Expression>,
    pub(crate) one_u32: Handle<Expression>,
    pub(crate) true_bool: Handle<Expression>,
    pub(crate) false_bool: Handle<Expression>,
    pub(crate) point_size_u32: Handle<Expression>,
}
impl Constants {
    fn new(func: &mut Function) -> Self {
        let mut lit = |l| func.expressions.add_unspanned(Expression::Literal(l));
        Self {
            zero_f32: lit(naga::Literal::F32(0.0)),
            one_f32: lit(naga::Literal::F32(1.0)),
            max_f32: lit(naga::Literal::F32(f32::MAX)),
            log2_max_f32: lit(naga::Literal::F32(f32::MAX.log2())),
            canonical_nan: lit(naga::Literal::U32(0b111111111 << 22)),
            zero_u32: lit(naga::Literal::U32(0)),
            one_u32: lit(naga::Literal::U32(1)),
            true_bool: lit(naga::Literal::U32(POINT_SIZE)),
            false_bool: lit(naga::Literal::Bool(true)),
            point_size_u32: lit(naga::Literal::Bool(false)),
        }
    }
}
#[derive(Debug)]
pub struct CompilingFunction {
    pub(crate) constants: Constants,
    pub(crate) invocation_id_u32: Handle<Expression>,
    pub(crate) frame_size_tallies: Vec<Option<Handle<Expression>>>,

    last_emit: usize,
    func: Function,
    // pointer to stack head
    pub(crate) stack_head: Handle<Expression>,
    // cache the current stack head when it hasn't been modified to avoid redundant loads
    stack_head_cache: Option<Handle<Expression>>,
    // pointer to stack buffer (ptr<device, [u32]>)
    pub(crate) stack_array_ptr: Handle<Expression>,
}
impl Deref for CompilingFunction {
    type Target = Function;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}
impl DerefMut for CompilingFunction {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.func
    }
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
            constants,
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

    /// inserts a function argument, returning a handle to its corresponding [`Expression::FunctionArgument`]
    fn add_arg(&mut self, arg: FunctionArgument) -> Handle<Expression> {
        let idx = self.arguments.len().try_into().unwrap();
        self.arguments.push(arg);
        self.add_preemit(Expression::FunctionArgument(idx))
    }
    fn flush_cache(&mut self) {
        self.stack_head_cache = None;
    }
    /// Switches the target for linear compilation to a new block, returning the old one.
    pub(crate) fn new_block(&mut self) -> Block {
        // barrier to make sure all expressions that were inserted in the old block get emitted within it
        self.emit_exprs();
        self.flush_cache();
        take(&mut self.body)
    }
    pub(crate) fn swap_block(&mut self, other: &mut Block) {
        self.emit_exprs();
        self.flush_cache();
        swap(&mut self.func.body, other);
    }
    #[track_caller]
    pub(crate) fn add_unspanned(&mut self, expr: naga::Expression) -> Handle<naga::Expression> {
        let ret = self.func.expressions.add_unspanned(expr);
        println!(
            "added expression {:#?}, location: {}",
            ret,
            Location::caller()
        );
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
        self.body
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
        println!(
            "emit({:#?}) called, location: {}",
            range.clone(),
            Location::caller()
        );
        self.body.push(Statement::Emit(range), Span::UNDEFINED);
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
            right: self.constants.one_u32,
        })
    }
    pub(crate) fn size_of_scalar(&self, ty: BaseType) -> Handle<Expression> {
        match ty {
            BaseType::Number => self.constants.one_u32,
            BaseType::Point => self.constants.point_size_u32,
            BaseType::Bool => todo!(),
            BaseType::Empty => todo!(),
            BaseType::Polygon => todo!(),
        }
    }
    #[track_caller]
    pub(crate) fn new_local(
        &mut self,
        ty: Handle<Type>,
        init: Option<Handle<Expression>>,
        name: Option<String>,
    ) -> Handle<Expression> {
        let local_handle = self
            .local_variables
            .add_unspanned(LocalVariable { name, ty, init });
        println!(
            "created local {:#?} at {}",
            local_handle,
            Location::caller()
        );
        self.add_preemit(Expression::LocalVariable(local_handle))
    }
    pub(crate) fn new_local_index(&mut self, ctx: &Compiler) -> Handle<Expression> {
        self.new_local(
            ctx.types.u32,
            Some(self.constants.zero_u32),
            Some(format!("iteration_var_{}", self.local_variables.len())),
        )
    }
    pub(crate) fn get_scalar_assignment(
        &mut self,
        ctx: &mut Compiler,
        id: usize,
        scalar: BaseType,
    ) -> Handle<naga::Expression> {
        println!("assignment: {id}, ty: {scalar:#?}");
        let s = ctx.scalar_type(scalar);
        if let crate::Assignment::Scalar(s) = ctx.assignments.entry(id).or_insert_with(|| {
            crate::Assignment::Scalar(self.new_local(
                s,
                None,
                Some(format!("scalar_assignment_{}", self.local_variables.len())),
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
                let scalar =
                    self.get_scalar_assignment(ctx, assignment.id, assignment.value.ty.base());

                self.push_frame();
                let value = compile_scalar(ctx, self, &assignment.value);
                self.pop_frame();

                self.store(scalar, value.inner);
            }
        }
    }
    pub(crate) fn done(mut self, ctx: &mut Compiler, ret: ScalarRef) -> Function {
        self.pop_frame();
        let s = self.size_of_scalar(ret.ty);
        let addr = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: s,
            right: self.invocation_id_u32,
        });
        let out_ptr = self.add_preemit(Expression::GlobalVariable(ctx.out_buf));
        self.store_to_array_typed(out_ptr, addr, ret);

        self.func
    }
    pub(crate) fn load_stack(&mut self, addr: Handle<Expression>) -> Handle<Expression> {
        let item_ptr = self.add_unspanned(Expression::Access {
            base: self.stack_array_ptr,
            index: addr,
        });
        self.load(item_ptr)
    }
    pub(crate) fn store_stack(&mut self, addr: Handle<Expression>, value: Handle<Expression>) {
        self.store_to_array(self.stack_array_ptr, addr, value);
    }
    pub(crate) fn store_to_array(
        &mut self,
        // ptr<array<u32>>
        base: Handle<Expression>,
        addr: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let item_ptr = self.add_unspanned(Expression::Access { base, index: addr });
        self.store(item_ptr, value);
    }

    pub(crate) fn store_to_top_of_stack_typed(&mut self, ctx: &Compiler, value: ScalarRef) {
        let s = self.size_of_scalar(value.ty);
        let base = self.bump_stack(ctx, s);
        self.store_stack_typed(base, value);
    }
    /// Converts an allocation and a raw index into a stack address
    pub(crate) fn index_alloc_to_addr(
        &mut self,
        alloc: &StackAlloc,
        idx: Handle<Expression>,
    ) -> Handle<Expression> {
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: alloc.base_addr,
            right: idx,
        })
    }
    /// computes an index's corresponding raw index (because points occupy multiple stack slots each)
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
    /// inverse of [`Self::index_to_raw`]
    pub(crate) fn raw_to_index(
        &mut self,
        raw: Handle<Expression>,
        ty: BaseType,
    ) -> Handle<Expression> {
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Divide,
            left: raw,
            right: self.size_of_scalar(ty),
        })
    }

    pub(crate) fn compute_stack_list_len(&mut self, r: &StackList) -> Handle<Expression> {
        let mut h = r.inner.len;
        if r.ty == BaseType::Point {
            h = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Divide,
                left: h,
                right: self.constants.point_size_u32,
            })
        }
        h
    }
    pub(crate) fn alloc_list(
        &mut self,
        ctx: &mut Compiler,
        ty: BaseType,
        mut len: Handle<Expression>,
    ) -> StackList {
        if ty == BaseType::Point {
            len = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: len,
                right: self.constants.point_size_u32,
            })
        }
        StackList::new(ty, self.alloc(ctx, len))
    }
    pub(crate) fn store_to_array_typed(
        &mut self,
        base_ptr: Handle<Expression>,
        mut addr: Handle<Expression>,
        value: ScalarRef,
    ) {
        match value.ty {
            BaseType::Number => {
                let h = self.bitcast_to_u32(value.inner);
                self.store_to_array(base_ptr, addr, h);
            }
            BaseType::Point => {
                let a = array::from_fn::<_, { POINT_SIZE as usize }, _>(|i| i as u32)
                    .map(|i| self.access_fixed(value.inner, i));
                for h in a {
                    let h = self.bitcast_to_u32(h);

                    self.store_to_array(base_ptr, addr, h);
                    addr = self.increment(addr);
                }
            }
            BaseType::Bool | BaseType::Empty | BaseType::Polygon => todo!(),
        }
    }
    pub(crate) fn store_stack_typed(&mut self, addr: Handle<Expression>, value: ScalarRef) {
        self.store_to_array_typed(self.stack_array_ptr, addr, value);
    }
    pub(crate) fn store_index_list_typed(
        &mut self,
        list: &StackList,
        index: Handle<Expression>,
        value: ScalarRef,
    ) {
        debug_assert!(list.types_eq(&value));

        let raw = self.index_to_raw(index, list.ty);
        let addr = self.index_alloc_to_addr(&list.inner, raw);
        self.store_stack_typed(addr, value);
    }

    pub(crate) fn load_stack_typed(
        &mut self,
        ctx: &Compiler,
        mut addr: Handle<Expression>,
        ty: BaseType,
    ) -> ScalarRef {
        ScalarRef::new(
            ty,
            match ty {
                BaseType::Number => {
                    let load = self.load_stack(addr);
                    self.bitcast_to_float(load)
                }
                BaseType::Point => {
                    let mut components = Vec::with_capacity(POINT_SIZE as usize);
                    for _ in 0..POINT_SIZE {
                        let val = self.load_stack(addr);

                        components.push(self.bitcast_to_float(val));

                        addr = self.increment(addr);
                    }
                    self.add_unspanned(Expression::Compose {
                        ty: ctx.types.point,
                        components,
                    })
                }
                BaseType::Bool | BaseType::Empty | BaseType::Polygon => todo!(),
            },
        )
    }
    pub(crate) fn load_typed_from_list(
        &mut self,
        ctx: &Compiler,
        list: &StackList,
        index: Handle<Expression>,
    ) -> ScalarRef {
        let raw = self.index_to_raw(index, list.ty);
        let addr = self.index_alloc_to_addr(&list.inner, raw);
        self.load_stack_typed(ctx, addr, list.ty)
    }

    /// Converts a Number value into a valid zero-indexed dynamic list index
    pub(crate) fn make_index(&mut self, number: Handle<Expression>) -> Handle<Expression> {
        let cast = self.add_unspanned(Expression::As {
            expr: number,
            kind: naga::ScalarKind::Uint,
            convert: Some(4),
        });
        // desmos is one-indexed :evilcat:
        self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Subtract,
            left: cast,
            right: self.constants.one_u32,
        })
    }
    pub(crate) fn u32_to_float(&mut self, number: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::As {
            expr: number,
            kind: naga::ScalarKind::Float,
            convert: Some(4),
        })
    }
}
