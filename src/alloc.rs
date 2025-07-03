use naga::{BinaryOperator, Expression, Handle, Literal};

use crate::{Compiler, function::CompilingFunction};
#[derive(Debug, Clone, Copy)]
pub(crate) struct StackAlloc {
    #[cfg(debug_assertions)]
    /// frame idx, generation
    stack_scope: (usize, usize),
    /// base address of the allocation
    pub base_addr: Handle<Expression>,
    /// length of this allocation in [`u32`]s.
    pub len: Handle<Expression>,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct StackState {
    frame_size_tallies: Vec<Handle<Expression>>,
    #[cfg(debug_assertions)]
    generation_counters: Vec<usize>,
}
impl StackState {
    pub(crate) fn new() -> Self {
        Default::default()
    }
    fn push_frame(&mut self, expr: Handle<Expression>) {
        self.frame_size_tallies.push(expr);
        #[cfg(debug_assertions)]
        {
            self.generation_counters
                .resize(self.frame_size_tallies.len(), 0);
        }
    }
    fn pop_frame(&mut self) -> Handle<Expression> {
        let ret = self.frame_size_tallies.pop().unwrap();
        #[cfg(debug_assertions)]
        {
            self.generation_counters[self.frame_size_tallies.len()] += 1;
        }
        ret
    }
    #[cfg(debug_assertions)]
    #[track_caller]
    fn current_scope(&self) -> (usize, usize) {
        let l = self.frame_size_tallies.len() - 1;
        (l, self.generation_counters[l])
    }
}

impl CompilingFunction {
    pub(crate) fn push_frame(&mut self, ctx: &mut Compiler) {
        ctx.stack.push_frame(self.zero_u32);
    }
    pub(crate) fn pop_frame(&mut self, ctx: &mut Compiler) {
        //self.emit_exprs();
        let pop_amount = ctx.stack.pop_frame();
        let load = self.load_stack_head();
        let new_val = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Subtract,
            left: load,
            right: pop_amount,
        });
        self.store_stack_head(new_val);
    }
    pub(crate) fn alloc(&mut self, ctx: &mut Compiler, req_len: Handle<Expression>) -> StackAlloc {
        let new_size = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Add,
            left: *ctx.stack.frame_size_tallies.last().unwrap(),
            right: req_len,
        });
        *ctx.stack.frame_size_tallies.last_mut().unwrap() = new_size;
        let read = self.load_stack_head();
        let new_val = self.add_unspanned(Expression::Binary {
            op: BinaryOperator::Add,
            left: read,
            right: req_len,
        });
        self.store_stack_head(new_val);
        #[cfg(debug_assertions)]
        StackAlloc {
            #[cfg(debug_assertions)]
            stack_scope: ctx.stack.current_scope(),
            base_addr: read,
            len: req_len,
        }
    }
    #[track_caller]
    // actual deallocation is performed by pop_frame, this is only for validating codegen
    pub(crate) fn dealloc(&mut self, ctx: &mut Compiler, a: StackAlloc) {
        #[cfg(debug_assertions)]
        {
            let (frame, generation) = a.stack_scope;
            assert!(frame < ctx.stack.frame_size_tallies.len());
            assert_eq!(generation, ctx.stack.generation_counters[frame])
        }
    }
}
