use std::{
    array,
    ops::{Deref, DerefMut},
};

use ambavia::type_checker::BaseType;
use naga::{
    Block, Expression, Function, FunctionArgument, FunctionResult, Handle, LocalVariable, Span,
    Statement,
};

use crate::{ArenaExt, Compiler, ScalarRef, StackList, alloc::StackAlloc};

const POINT_SIZE: u32 = 2;

#[derive(Debug)]
pub struct CompilingFunction {
    one_u32: Handle<Expression>,
    last_emit: usize,
    func: Function,
    // pointer to stack head
    pub(crate) stack_head: Handle<Expression>,
    // pointer to stack buffer (ptr<device, [u32]>)
    pub(crate) stack_ptr: Handle<Expression>,
    block: Block,
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
    pub(crate) fn new_primary(ctx: &mut Compiler) -> Self {
        let mut func = Function {
            name: Some("capuronii_main".into()),
            arguments: vec![
                FunctionArgument {
                    name: "invocation_id".to_string().into(),
                    ty: ctx.ty_ctx.uvec3,
                    binding: None,
                },
                FunctionArgument {
                    name: "num_workgroups".to_string().into(),
                    ty: ctx.ty_ctx.uvec3,
                    binding: None,
                },
            ],
            result: Some(FunctionResult {
                ty: ctx.ty_ctx.f32,
                binding: None,
            }),
            ..Default::default()
        };

        let inv_id = func
            .expressions
            .add_unspanned(naga::Expression::FunctionArgument(0));
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
        let stack_base = gen_local("stack_base", ctx.ty_ctx.u32, None);
        let stack_head = gen_local("stack_head", ctx.ty_ctx.u32, None);

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
        let (num_x, num_y, num_z) = (
            func.add_unspanned(naga::Expression::AccessIndex {
                base: num_workgroups,
                index: 0,
            }),
            func.add_unspanned(naga::Expression::AccessIndex {
                base: num_workgroups,
                index: 1,
            }),
            func.add_unspanned(naga::Expression::AccessIndex {
                base: num_workgroups,
                index: 2,
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
        let heap_per_invocation = func.add_preemit(naga::Expression::Literal(naga::Literal::U32(
            ctx.heap_per_invocation,
        )));
        let heap_offset = naga::Expression::Binary {
            op: naga::BinaryOperator::Multiply,
            left: flattened_inv_id,
            right: heap_per_invocation,
        };
        func.store_local(stack_base, heap_offset.clone());
        func.store_local(stack_head, heap_offset);
        func.push_frame(ctx);
        func
    }
    /// inserts a function argument, returning a handle to its corresponding [`Expression::FunctionArgument`]
    fn add_arg(&mut self, arg: FunctionArgument) -> Handle<Expression> {
        let idx = self.arguments.len().try_into().unwrap();
        self.arguments.push(arg);
        self.add_preemit(Expression::FunctionArgument(idx))
    }
    fn new(
        mut func: Function,
        stack_head: Handle<Expression>,
        stack_ptr: Handle<Expression>,
    ) -> Self {
        let one_u32 = func
            .expressions
            .add_unspanned(Expression::Literal(naga::Literal::U32(1)));
        Self {
            last_emit: 0,
            func,
            stack_head,
            block: Block::new(),
            stack_ptr,
            one_u32,
        }
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
    pub(crate) fn add_unspanned(&mut self, expr: naga::Expression) -> Handle<naga::Expression> {
        self.func.expressions.add_unspanned(expr)
    }
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
    pub(crate) fn store_local(&mut self, pointer: Handle<Expression>, value: Expression) {
        let value = self.add_unspanned(value);
        self.emit_exprs();
        self.block
            .push(Statement::Store { pointer, value }, Span::UNDEFINED);
    }
    /// stores a raw value (uint/u32) to a raw offset within a stack allocation
    pub(crate) fn store_to_stack(
        &mut self,
        alloc: &StackAlloc,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let offset_idx = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: alloc.base_addr,
            right: index,
        });
        let head = self.load(self.stack_head);
        let offset = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: head,
            right: offset_idx,
        });
        let item_ptr = self.add_unspanned(Expression::Access {
            base: self.stack_ptr,
            index: offset,
        });
        self.emit_exprs();
        self.block.push(
            Statement::Store {
                pointer: item_ptr,
                value,
            },
            Span::UNDEFINED,
        );
    }
    /// loads a raw value (uint/u32) from a raw offset within a stack allocation
    pub(crate) fn load_from_stack(
        &mut self,
        alloc: &StackAlloc,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let offset_idx = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: alloc.base_addr,
            right: index,
        });
        let head = self.load(self.stack_head);
        let offset = self.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: head,
            right: offset_idx,
        });
        let item_ptr = self.add_unspanned(Expression::Access {
            base: self.stack_ptr,
            index: offset,
        });
        self.add_unspanned(Expression::Load { pointer: item_ptr })
    }

    pub(crate) fn skip_emit_exprs(&mut self) {
        self.last_emit = self.func.expressions.len();
    }
    pub(crate) fn emit_exprs(&mut self) {
        self.block.push(
            Statement::Emit(self.func.expressions.range_from(self.last_emit)),
            Span::UNDEFINED,
        );
        self.last_emit = self.func.expressions.len();
    }
    pub(crate) fn new_scalar_assignment(
        &mut self,
        ctx: &mut Compiler,
        id: usize,
        scalar: BaseType,
    ) -> Handle<naga::Expression> {
        // todo allow mapping assignment name
        let local_handle = self.local_variables.add_unspanned(LocalVariable {
            name: None,
            ty: ctx.map_scalar(scalar),
            init: None,
        });
        let x = self.add_preemit(Expression::LocalVariable(local_handle));
        ctx.scalar_assignments.insert(id, x);
        x
    }
    pub(crate) fn done(mut self, ctx: &mut Compiler) -> Function {
        self.pop_frame(ctx);
        self.func.body = self.block;
        self.func
    }
    pub(crate) fn compute_list_len(&mut self, r: StackList) -> Handle<Expression> {
        let mut h = r.inner.len;
        if r.ty == BaseType::Point {
            let two = self.add_preemit(Expression::Literal(naga::Literal::U32(POINT_SIZE)));
            h = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Divide,
                left: h,
                right: two,
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
            let two = self.add_preemit(Expression::Literal(naga::Literal::U32(POINT_SIZE)));
            len = self.add_unspanned(Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: len,
                right: two,
            })
        }
        StackList::new(ty, self.alloc(ctx, len))
    }
    pub(crate) fn store_typed(
        &mut self,
        list: &StackList,
        mut index: Handle<Expression>,
        value: ScalarRef,
    ) {
        debug_assert!(list.types_eq(&value));
        match value.ty {
            BaseType::Number => {
                let h = self.bitcast_to_u32(value.inner);
                self.store_to_stack(&list.inner, index, h);
            }
            BaseType::Point => {
                let two = self.add_preemit(Expression::Literal(naga::Literal::U32(POINT_SIZE)));
                index = self.add_unspanned(Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    left: index,
                    right: two,
                });
                let a = array::from_fn::<_, { POINT_SIZE as usize }, _>(|i| i as u32)
                    .map(|i| self.access_fixed(value.inner, i));

                for h in a {
                    let h = self.bitcast_to_u32(h);
                    self.store_to_stack(&list.inner, index, h);
                    index = self.increment(index);
                }
            }
            BaseType::Bool | BaseType::Empty => todo!(),
        }
    }
    pub(crate) fn load_typed(
        &mut self,
        ctx: &Compiler,
        list: &StackList,
        mut index: Handle<Expression>,
    ) -> ScalarRef {
        ScalarRef::new(
            list.ty,
            match list.ty {
                BaseType::Number => {
                    let v = self.load_from_stack(&list.inner, index);
                    self.bitcast_to_float(v)
                }
                BaseType::Point => {
                    let size =
                        self.add_preemit(Expression::Literal(naga::Literal::U32(POINT_SIZE)));
                    index = self.add_unspanned(Expression::Binary {
                        op: naga::BinaryOperator::Multiply,
                        left: index,
                        right: size,
                    });
                    let mut components = Vec::with_capacity(POINT_SIZE as usize);
                    for _ in 0..POINT_SIZE {
                        let val = self.load_from_stack(&list.inner, index);
                        components.push(self.bitcast_to_float(val));
                        index = self.increment(index);
                    }
                    self.add_unspanned(Expression::Compose {
                        ty: ctx.ty_ctx.point,
                        components,
                    })
                }
                BaseType::Bool | BaseType::Empty => todo!(),
            },
        )
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
            right: self.one_u32,
        })
    }
    pub(crate) fn make_index(&mut self, number: Handle<Expression>) -> Handle<Expression> {
        self.add_unspanned(Expression::As {
            expr: number,
            kind: naga::ScalarKind::Uint,
            convert: Some(4),
        })
    }
}
