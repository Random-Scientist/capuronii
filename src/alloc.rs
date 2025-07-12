use naga::{BinaryOperator, Expression, Handle, Literal};

use crate::{Compiler, function::CompilingFunction};
#[derive(Debug, Clone, Copy)]
pub struct StackAlloc {
    /// base address of the allocation
    pub base_addr: Handle<Expression>,
    /// length of this allocation in [`u32`]s.
    pub len: Handle<Expression>,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct StackState {
    frame_size_tallies: Vec<Handle<Expression>>,
}
impl StackState {
    pub(crate) fn new() -> Self {
        Default::default()
    }
    fn push_frame(&mut self, expr: Handle<Expression>) {
        self.frame_size_tallies.push(expr);
    }
    fn pop_frame(&mut self) -> Handle<Expression> {
        self.frame_size_tallies.pop().unwrap()
    }
    fn current(&self) -> Handle<Expression> {
        *self.frame_size_tallies.last().unwrap()
    }
}

impl CompilingFunction {
    pub(crate) fn push_frame(&mut self, ctx: &mut Compiler) {
        let f = self.new_local(ctx.types.u32, Some(self.zero_u32));
        ctx.stack.push_frame(f);
    }

    pub(crate) fn pop_frame(&mut self, ctx: &mut Compiler) {
        let pop_amount = self.load(ctx.stack.pop_frame());

        let load = self.load_stack_head();
        let new_val = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Subtract,
            left: load,
            right: pop_amount,
        });
        self.store_stack_head(new_val);
    }
    pub(crate) fn alloc(&mut self, ctx: &mut Compiler, req_size: Handle<Expression>) -> StackAlloc {
        let frame_size = ctx.stack.current();
        let prev_size = self.load(frame_size);
        let new_size = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Add,
            left: prev_size,
            right: req_size,
        });
        self.store(frame_size, new_size);

        let read = self.load_stack_head();
        let new_val = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Add,
            left: read,
            right: req_size,
        });
        self.store_stack_head(new_val);
        StackAlloc {
            base_addr: read,
            len: req_size,
        }
    }
}
