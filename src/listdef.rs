use std::num::NonZeroU32;

use naga::{Block, Handle};
use parse::type_checker::{self, Assignment, BaseType};
use replace_with::{replace_with, replace_with_or_abort};

use crate::{
    ArenaExt, Compiler, ScalarRef, WithScalarType, alloc::StackAlloc, compile_scalar,
    function::CompilingFunction,
};

pub struct LazyBroadcast<'a> {
    pub(crate) varying: Vec<(usize, ListDef<'a>)>,
    pub(crate) body: &'a type_checker::TypedExpression,
    pub(crate) scalars: &'a [Assignment],
}
impl LazyBroadcast<'_> {
    fn compute_len(
        &mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> Handle<naga::Expression> {
        let mut it = self.varying.iter_mut();
        let mut len = it.next().unwrap().1.compute_len(ctx, func);
        for i in it {
            let next = i.1.compute_len(ctx, func);
            len = func.add_unspanned(naga::Expression::Math {
                fun: naga::MathFunction::Min,
                arg: len,
                arg1: Some(next),
                arg2: None,
                arg3: None,
            });
        }
        len
    }

    fn index(
        self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
        idx: Handle<naga::Expression>,
    ) -> ScalarRef {
        // bind varyings
        for (id, list) in self.varying {
            let assignment = func.get_scalar_assignment(ctx, id, list.ty);
            let value = list.index(idx, ctx, func);
            func.store(assignment, value);
        }
        func.push_frame(ctx);
        let r = compile_scalar(ctx, func, self.body);
        func.pop_frame(ctx);
        r
    }
}
pub struct LazyComprehension<'a> {
    pub(crate) varying: Vec<(usize, ListDef<'a>)>,
    pub(crate) body: &'a type_checker::Body,
}
impl LazyComprehension<'_> {
    fn compute_len(
        &mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> Handle<naga::Expression> {
        let mut it = self.varying.iter_mut();
        let mut len = it.next().unwrap().1.compute_len(ctx, func);
        for i in it {
            let next = i.1.compute_len(ctx, func);
            len = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: len,
                right: next,
            });
        }
        len
    }
    fn index(
        mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
        idx: Handle<naga::Expression>,
    ) -> Handle<naga::Expression> {
        let mut lengths: Vec<_> = self
            .varying
            .iter_mut()
            .map(|(id, l)| l.compute_len(ctx, func))
            .collect();

        todo!()
    }
}

pub struct LazyStaticList<'a> {
    pub(crate) elements: &'a [type_checker::TypedExpression],
}

pub struct StaticList {
    // handle to array
    pub(crate) len: Handle<naga::Expression>,
    // ptr to [u32; len]
    pub(crate) arr: Handle<naga::Expression>,
}
pub struct Join<'a>(Vec<ListDef<'a>>);
pub struct Filter<'a> {
    pub src: Box<ListDef<'a>>,
    // produces type bool
    pub filter: LazyBroadcast<'a>,
}
impl Filter<'_> {
    fn materialize(mut self, ctx: &mut Compiler, func: &mut CompilingFunction) -> MaterializedList {
        let filter_len = self.filter.compute_len(ctx, func);
        let src_len = self.src.compute_len(ctx, func);
        let max_len = func.add_unspanned(naga::Expression::Math {
            fun: naga::MathFunction::Min,
            arg: filter_len,
            arg1: Some(src_len.0),
            arg2: None,
            arg3: None,
        });

        let out_ty = self.src.ty;
        let iter_index = func.new_local(ctx.types.u32, Some(func.zero_u32));
        let out_len = func.new_local(ctx.types.u32, Some(func.zero_u32));

        let out_base_addr = func.load_stack_head();

        // loop body block
        let mut function_body = func.new_block();
        let this_iter_index = func.load(iter_index);
        let test = self.filter.index(ctx, func, this_iter_index);

        let mut loop_block = func.new_block();
        let result = self.src.index(this_iter_index, ctx, func);
        func.store_to_top_of_stack_typed(ScalarRef::new(out_ty, result));

        let current_out_len = func.load(out_len);

        let s = func.size_of_scalar(out_ty);
        let inced = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Add,
            left: current_out_len,
            right: s,
        });

        func.store(out_len, inced);

        func.swap_block(&mut loop_block);
        let success_case = loop_block;
        func.body.push(
            naga::Statement::If {
                condition: test.inner,
                accept: success_case,
                reject: Block::new(),
            },
            naga::Span::UNDEFINED,
        );
        let next_iter_index = func.increment(this_iter_index);
        func.emit_exprs();

        let should_break = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Greater,
            left: next_iter_index,
            right: max_len,
        });
        func.body.push(
            naga::Statement::If {
                condition: should_break,
                accept: Block::from_vec(vec![naga::Statement::Break]),
                reject: Block::new(),
            },
            naga::Span::UNDEFINED,
        );
        func.store(iter_index, next_iter_index);

        func.swap_block(&mut function_body);
        let loop_body = function_body;
        func.body.push(
            naga::Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            naga::Span::UNDEFINED,
        );
        MaterializedList::Temporary(StackAlloc {
            base_addr: out_base_addr,
            len: out_len,
        })
    }
}
pub enum UntypedListDef<'a> {
    Materialized(MaterializedList),
    Broadcast(LazyBroadcast<'a>),
    Comprehension(LazyComprehension<'a>),
    LazyStatic(LazyStaticList<'a>),
    Join(Join<'a>),
    Filter(Filter<'a>),
}

pub enum MaterializedList {
    Temporary(StackAlloc),
    Static(StaticList),
    Empty,
}

trait MaybeSwitchBlock {
    fn get(&mut self) -> &mut CompilingFunction;
    fn swap(&mut self) {}
}
impl MaybeSwitchBlock for CompilingFunction {
    fn get(&mut self) -> &mut CompilingFunction {
        self
    }
}

pub struct WithOtherBlock {
    other: Block,
    func: CompilingFunction,
}
impl MaybeSwitchBlock for WithOtherBlock {
    fn get(&mut self) -> &mut CompilingFunction {
        &mut self.func
    }
    fn swap(&mut self) {
        self.func.swap_block(&mut self.other);
    }
}
impl ListDef<'_> {
    fn compute_len(
        &mut self,
        ctx: &mut Compiler,
        func: &mut impl MaybeSwitchBlock,
    ) -> Handle<naga::Expression> {
        match &mut self.inner {
            UntypedListDef::Materialized(materialized_list) => match materialized_list {
                MaterializedList::Temporary(stack_alloc) => {
                    func.compute_stack_list_len(&WithScalarType {
                        ty: self.ty,
                        inner: *stack_alloc,
                    })
                }
                MaterializedList::Static(static_list) => static_list.len,
                MaterializedList::Empty => func.zero_u32,
            },
            UntypedListDef::Broadcast(lazy_broadcast) => lazy_broadcast.compute_len(ctx, func),
            UntypedListDef::Comprehension(lazy_comprehension) => {
                lazy_comprehension.compute_len(ctx, func)
            }
            UntypedListDef::LazyStatic(lazy_static_list) => {
                func.add_preemit(naga::Expression::Literal(naga::Literal::U32(
                    lazy_static_list.elements.len() as u32,
                )))
            }
            UntypedListDef::Join(join) => {
                let mut it = join.0.iter_mut();
                let mut len = it.next().unwrap().compute_len(ctx, func);
                for i in it {
                    let next = i.compute_len(ctx, func);
                    len = func.add_unspanned(naga::Expression::Binary {
                        op: naga::BinaryOperator::Add,
                        left: len,
                        right: next,
                    });
                }
                len
            }
            UntypedListDef::Filter(_) => {
                replace_with_or_abort(&mut self.inner, |v| {
                    let UntypedListDef::Filter(f) = v else {
                        panic!()
                    };
                    UntypedListDef::Materialized(f.materialize(ctx, func))
                });
                self.compute_len(ctx, func)
            }
        }
    }
    pub(crate) fn index(
        mut self,
        idx: Handle<naga::Expression>,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> Handle<naga::Expression> {
        match self.inner {
            UntypedListDef::Materialized(materialized_list) => {
                materialized_list.index(ctx, func, self.ty, idx)
            }
            UntypedListDef::Broadcast(lazy_broadcast) => lazy_broadcast.index(ctx, func, idx).inner,
            UntypedListDef::Comprehension(lazy_comprehension) => unreachable!(),
            UntypedListDef::LazyStatic(LazyStaticList { elements }) => {
                //TODO decide when to allocate to stack buffer to avoid overflowing/bloating the GPU stack with massive local arrays
                self.inner = if elements.is_empty() {
                    UntypedListDef::Materialized(MaterializedList::Empty)
                } else {
                    let new_arr_ty = ctx.module.types.add_unspanned(naga::Type {
                        name: None,
                        inner: naga::TypeInner::Array {
                            base: ctx.scalar_type(self.ty),
                            size: naga::ArraySize::Constant(
                                NonZeroU32::new(elements.len() as u32).unwrap(),
                            ),
                            stride: 4,
                        },
                    });
                    let components = elements
                        .iter()
                        .map(|e| {
                            func.push_frame(ctx);
                            let r = compile_scalar(ctx, func, e).inner;
                            func.pop_frame(ctx);
                            r
                        })
                        .collect();
                    let arr = func.add_unspanned(naga::Expression::Compose {
                        ty: new_arr_ty,
                        components,
                    });
                    let len = func.add_preemit(naga::Expression::Literal(naga::Literal::U32(
                        elements.len() as u32,
                    )));
                    UntypedListDef::Materialized(MaterializedList::Static(StaticList { len, arr }))
                };
                self.index(idx, ctx, func)
            }
            UntypedListDef::Join(join) => {}
            UntypedListDef::Filter(filter) => todo!(),
        }
    }
    pub(crate) fn materialize(
        self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> MaterializedList {
        todo!()
    }
}
impl MaterializedList {
    fn index(
        &self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
        ty: BaseType,
        index: Handle<naga::Expression>,
    ) -> Handle<naga::Expression> {
        match self {
            MaterializedList::Temporary(stack_alloc) => {
                let raw = func.index_to_raw(index, ty);
                let addr = func.index_alloc_to_addr(stack_alloc, raw);
                func.load_stack_typed(ctx, addr, ty).inner
            }
            MaterializedList::Static(static_list) => func.add_unspanned(naga::Expression::Access {
                base: static_list.arr,
                index,
            }),
            // when GPU panics are implemented this should be one
            MaterializedList::Empty => {
                func.add_preemit(naga::Expression::ZeroValue(ctx.scalar_type(ty)))
            }
        }
    }
}
pub type ListDef<'a> = WithScalarType<UntypedListDef<'a>>;
pub type StackList = WithScalarType<StackAlloc>;
