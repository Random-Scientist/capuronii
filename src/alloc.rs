use naga::{BinaryOperator, Expression, Handle};

use crate::{Compiler, function::CompilingFunction};
#[derive(Debug, Clone, Copy)]
pub struct StackAlloc {
    /// base address of this allocation within the above array
    pub base_addr: Handle<Expression>,
    /// (runtime) length of this allocation in [`u32`]s.
    pub size: Handle<Expression>,
}
#[derive(Debug, Clone, Copy)]
pub struct StaticAlloc {
    // pointer to an array of u32s.
    pub array_pointer: Handle<Expression>,
    /// (compile time) length of this allocation in [`u32`]s.
    pub size: u32,
}
#[derive(Debug, Clone, Copy)]
pub enum AllocSize {
    Dynamic(Handle<Expression>),
    Static(u32),
}
pub(crate) trait Allocation: Sized + Copy {
    fn load_at(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
    ) -> Handle<Expression>;
    fn store_to(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
        value: Handle<Expression>,
    );
    fn size(self) -> AllocSize;
}
impl Allocation for StackAlloc {
    fn load_at(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
    ) -> Handle<Expression> {
        let addr = func.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: self.base_addr,
            right: offset,
        });
        let pointer = func.access(func.stack_array_ptr, addr);
        func.load(pointer)
    }

    fn store_to(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let addr = func.add_unspanned(Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: self.base_addr,
            right: offset,
        });
        let pointer = func.access(func.stack_array_ptr, addr);
        func.store(pointer, value);
    }
    fn size(self) -> AllocSize {
        AllocSize::Dynamic(self.size)
    }
}
impl Allocation for StaticAlloc {
    fn load_at(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
    ) -> Handle<Expression> {
        let pointer = func.access(self.array_pointer, offset);
        func.load(pointer)
    }

    fn store_to(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = func.access(self.array_pointer, offset);
        func.store(pointer, value)
    }

    fn size(self) -> AllocSize {
        AllocSize::Static(self.size)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Alloc {
    Dynamic(StackAlloc),
    Static(StaticAlloc),
}
impl From<StackAlloc> for Alloc {
    fn from(value: StackAlloc) -> Self {
        Self::Dynamic(value)
    }
}
impl From<StaticAlloc> for Alloc {
    fn from(value: StaticAlloc) -> Self {
        Self::Static(value)
    }
}

impl Allocation for Alloc {
    fn load_at(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
    ) -> Handle<Expression> {
        match self {
            Alloc::Dynamic(stack_alloc) => stack_alloc.load_at(func, offset),
            Alloc::Static(static_alloc) => static_alloc.load_at(func, offset),
        }
    }

    fn store_to(
        self,
        func: &mut CompilingFunction,
        offset: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        match self {
            Alloc::Dynamic(stack_alloc) => stack_alloc.store_to(func, offset, value),
            Alloc::Static(static_alloc) => static_alloc.store_to(func, offset, value),
        }
    }

    fn size(self) -> AllocSize {
        match self {
            Alloc::Dynamic(stack_alloc) => stack_alloc.size(),
            Alloc::Static(static_alloc) => static_alloc.size(),
        }
    }
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
                Some(self.consts.zero_u32),
                Some(format!("stack_frame_{}", self.frame_size_tallies.len())),
            );
            *self.frame_size_tallies.last_mut().unwrap() = Some(local);
            local
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
    pub(crate) fn alloc_stack(&mut self, ctx: &Compiler, req_size: Handle<Expression>) -> Alloc {
        StackAlloc {
            base_addr: self.bump_stack(ctx, req_size),
            size: req_size,
        }
        .into()
    }
}
