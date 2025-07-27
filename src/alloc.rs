use naga::{BinaryOperator, Expression, Handle};

use crate::{Compiler, function::CompilingFunction};
#[derive(Debug, Clone, Copy)]
pub struct StackAlloc {
    /// base address of the allocation
    pub base_addr: Handle<Expression>,
    /// length of this allocation in [`u32`]s.
    pub len: Handle<Expression>,
}

impl CompilingFunction {
    pub(crate) fn push_frame(&mut self) {
        self.frame_size_tallies.push(None);
    }

    pub(crate) fn pop_frame(&mut self) {
        let Some(to_pop) = self.frame_size_tallies.pop().unwrap() else {
            return;
        };
        let pop_amount = self.load(to_pop);

        let load = self.load_stack_head();
        let new_val = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Subtract,
            left: load,
            right: pop_amount,
        });
        self.store_stack_head(new_val);
    }
    fn materialize_top_frame_size(&mut self, ctx: &Compiler) -> Handle<Expression> {
        if let Some(tally) = self.frame_size_tallies.last_mut().unwrap() {
            *tally
        } else {
            let local = self.new_local(
                ctx.types.u32,
                Some(self.constants.zero_u32),
                Some(format!("stack_frame_{}", self.frame_size_tallies.len())),
            );
            *self.frame_size_tallies.last_mut().unwrap() = Some(local);
            local
        }
    }
    pub(crate) fn alloc(&mut self, ctx: &mut Compiler, req_size: Handle<Expression>) -> StackAlloc {
        StackAlloc {
            base_addr: self.bump_stack(ctx, req_size),
            len: req_size,
        }
    }
    pub(crate) fn bump_stack(
        &mut self,
        ctx: &Compiler,
        size: Handle<Expression>,
    ) -> Handle<Expression> {
        let frame_size = self.materialize_top_frame_size(ctx);

        let prev_size = self.load(frame_size);

        let new_size = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Add,
            left: prev_size,
            right: size,
        });
        self.store(frame_size, new_size);

        let read = self.load_stack_head();
        let new_val = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Add,
            left: read,
            right: size,
        });
        self.store_stack_head(new_val);
        read
    }
}
