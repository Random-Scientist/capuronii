use std::num::NonZeroU32;

use naga::{Block, Handle, Statement};
use parse::type_checker::{self, Assignment, BaseType};
use replace_with::replace_with_or_abort;

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
        let mut len = it.next().unwrap().1.compute_len_inner(ctx, func);
        for i in it {
            let next = i.1.compute_len_inner(ctx, func);
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
        idx: Handle<naga::Expression>,
        ctx: &mut Compiler,
        func: &mut impl MaybeSwitchBlock,
    ) -> Handle<naga::Expression> {
        let p = func.in_outer();
        p.bind_assignments(ctx, self.scalars);

        // bind varyings
        for (id, list) in self.varying {
            let assignment = func.in_inner().get_scalar_assignment(ctx, id, list.ty);
            let value = list.index(idx, ctx, func);

            func.in_inner().store(assignment, value);
        }
        let eval_in = func.in_inner();
        eval_in.push_frame();
        let r = compile_scalar(ctx, eval_in, self.body);
        eval_in.pop_frame();
        r.inner
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
        let mut len = it.next().unwrap().1.compute_len_inner(ctx, func);
        for i in it {
            let next = i.1.compute_len_inner(ctx, func);
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
        idx: Handle<naga::Expression>,
        ctx: &mut Compiler,
        blocks: &mut impl MaybeSwitchBlock,
    ) -> Handle<naga::Expression> {
        let pre = blocks.in_outer();
        let lengths: Vec<_> = self
            .varying
            .iter_mut()
            .map(|(id, l)| l.compute_len_inner(ctx, pre))
            .collect();
        // Determine and index comprehension varyings
        // TODO make this only update a given varying binding when its index changes
        let mut prev_section_len = pre.constants.one_u32;
        for ((assignment, value), length) in self.varying.into_iter().zip(lengths) {
            let out_ty = value.ty;
            let func = blocks.in_inner();
            // integer division truncates towards zero, which is equivalent to `div_floor` (the desired primitive here) for whole number values
            let div = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::Divide,
                left: idx,
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
            let val = value.index(index, ctx, blocks);

            let func = blocks.in_inner();
            let s = func.get_scalar_assignment(ctx, assignment, out_ty);
            func.store(s, val);
        }
        let func = blocks.in_inner();

        // frame for listcomp deps

        func.push_frame();
        func.bind_assignments(ctx, &self.body.assignments);
        let result = compile_scalar(ctx, func, &self.body.value);
        func.pop_frame();

        result.inner
    }
}

pub struct LazyStaticList<'a> {
    pub(crate) elements: &'a [type_checker::TypedExpression],
}

pub struct StaticList {
    // u32
    pub(crate) len: Handle<naga::Expression>,
    // ptr to [u32; len]
    pub(crate) arr: Handle<naga::Expression>,
}
pub struct Join<'a> {
    pub lists: Vec<ListDef<'a>>,
}
impl Join<'_> {
    fn compute_len(
        &mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> Handle<naga::Expression> {
        let mut it = self.lists.iter_mut();
        let mut len = it.next().unwrap().compute_len_inner(ctx, func);
        for i in it {
            let next = i.compute_len_inner(ctx, func);
            len = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: len,
                right: next,
            });
        }
        len
    }
    fn index(
        self,
        idx: Handle<naga::Expression>,
        ctx: &mut Compiler,
        blocks: &mut impl MaybeSwitchBlock,
    ) -> Handle<naga::Expression> {
        let pre = blocks.in_outer();
        let mut curr_start_idx = pre.constants.zero_u32;

        let val_out = pre.new_local(
            ctx.scalar_type(self.lists.first().unwrap().ty),
            None,
            Some(format!("join_output_{}", pre.local_variables.len())),
        );

        let mut body_block = blocks.in_inner().new_block();

        let mut cond_blocks = Vec::with_capacity(self.lists.len());

        for mut list in self.lists.into_iter().rev() {
            let pre = blocks.in_outer();
            let this_len = list.compute_len_inner(ctx, pre);

            let func = blocks.in_inner();
            let test = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::GreaterEqual,
                left: idx,
                right: curr_start_idx,
            });
            let test_block = func.new_block();

            let transformed_idx = func.add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::Subtract,
                left: idx,
                right: curr_start_idx,
            });

            let val = list.index(transformed_idx, ctx, blocks);
            let func = blocks.in_inner();
            func.store(val_out, val);

            let success_block = func.new_block();

            cond_blocks.push((test, test_block, success_block));

            curr_start_idx = blocks.in_outer().add_unspanned(naga::Expression::Binary {
                op: naga::BinaryOperator::Add,
                left: curr_start_idx,
                right: this_len,
            });
        }
        let nested_if = cond_blocks.into_iter().fold(
            Block::default(),
            |rest, (test, mut test_block, success)| {
                test_block.push(
                    naga::Statement::If {
                        condition: test,
                        accept: success,
                        reject: rest,
                    },
                    naga::Span::UNDEFINED,
                );
                test_block
            },
        );
        let func = blocks.in_inner();

        func.swap_block(&mut body_block);
        func.body
            .push(naga::Statement::Block(nested_if), naga::Span::UNDEFINED);

        func.load(val_out)
    }
}
pub struct Filter<'a> {
    pub src: Box<ListDef<'a>>,
    // produces type bool
    pub filter: LazyBroadcast<'a>,
}
impl Filter<'_> {
    // TODO reuse allocation when the source is already a materialized list
    fn materialize(mut self, ctx: &mut Compiler, func: &mut CompilingFunction) -> MaterializedList {
        let filter_len = self.filter.compute_len(ctx, func);
        let src_len = self.src.compute_len_inner(ctx, func);

        let max_len = func.add_unspanned(naga::Expression::Math {
            fun: naga::MathFunction::Min,
            arg: filter_len,
            arg1: Some(src_len),
            arg2: None,
            arg3: None,
        });

        let out_ty = self.src.ty;
        let iter_index = func.new_local_index(ctx);
        let out_len = func.new_local_index(ctx);

        let out_base_addr = func.load_stack_head();

        // loop body block
        let function_body = func.new_block();

        let this_iter_index = func.load(iter_index);

        let mut other = WithEvalBlock {
            other: function_body,
            func,
            in_eval: true,
        };

        let test = self.filter.index(this_iter_index, ctx, &mut other);

        let mut loop_block = other.in_inner().new_block();
        let result = self.src.index(this_iter_index, ctx, &mut other);

        let func = other.in_inner();

        func.store_to_top_of_stack_typed(ctx, ScalarRef::new(out_ty, result));

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
                condition: test,
                accept: success_case,
                reject: Block::new(),
            },
            naga::Span::UNDEFINED,
        );
        let next_iter_index = func.increment(this_iter_index);

        let should_break = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Greater,
            left: next_iter_index,
            right: max_len,
        });
        func.emit_exprs();
        func.body.push(
            naga::Statement::If {
                condition: should_break,
                accept: Block::from_vec(vec![naga::Statement::Break]),
                reject: Block::new(),
            },
            naga::Span::UNDEFINED,
        );
        func.store(iter_index, next_iter_index);
        // ensure func contains the primary body block, not the loop body block
        let _ = other.in_outer();
        let WithEvalBlock {
            other: body, func, ..
        } = other;

        func.body.push(
            naga::Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            naga::Span::UNDEFINED,
        );
        let len = func.load(out_len);

        MaterializedList::Temporary(StackAlloc {
            base_addr: out_base_addr,
            len,
        })
    }
}

pub struct Select<'a> {
    pub test: &'a type_checker::TypedExpression,
    pub consequent_alternate: Box<(ListDef<'a>, ListDef<'a>)>,
}
pub enum UntypedListDef<'a> {
    Materialized(MaterializedList),
    Broadcast(LazyBroadcast<'a>),
    Comprehension(LazyComprehension<'a>),
    LazyStatic(LazyStaticList<'a>),
    Join(Join<'a>),
    Filter(Filter<'a>),
    Select(Select<'a>),
}
pub struct ListDef<'a> {
    pub ty: BaseType,
    pub inner: UntypedListDef<'a>,
    cached_len: Option<Handle<naga::Expression>>,
    cached_select_test: Option<Handle<naga::Expression>>,
}
impl<'a> ListDef<'a> {
    pub fn new(ty: BaseType, inner: UntypedListDef<'a>) -> Self {
        Self {
            ty,
            inner,
            cached_len: None,
            cached_select_test: None,
        }
    }
}
pub enum MaterializedList {
    Temporary(StackAlloc),
    Static(StaticList),
    Empty,
}

pub(crate) trait MaybeSwitchBlock: Sized {
    fn in_outer(&mut self) -> &mut CompilingFunction;
    fn in_inner(&mut self) -> &mut CompilingFunction {
        // default implementation does eval and prelude in the same block
        self.in_outer()
    }
    fn into_inner<'a>(self) -> (&'a mut CompilingFunction, Option<Block>)
    where
        Self: 'a;
}
impl MaybeSwitchBlock for &'_ mut CompilingFunction {
    fn in_outer(&mut self) -> &mut CompilingFunction {
        self
    }

    fn into_inner<'a>(self) -> (&'a mut CompilingFunction, Option<Block>)
    where
        Self: 'a,
    {
        (self, None)
    }
}

pub struct WithEvalBlock<'a> {
    other: Block,
    func: &'a mut CompilingFunction,
    in_eval: bool,
}
impl MaybeSwitchBlock for WithEvalBlock<'_> {
    fn in_outer(&mut self) -> &mut CompilingFunction {
        if self.in_eval {
            self.func.swap_block(&mut self.other);
            self.in_eval = false;
        }
        self.func
    }
    fn in_inner(&mut self) -> &mut CompilingFunction {
        if !self.in_eval {
            self.func.swap_block(&mut self.other);
            self.in_eval = true;
        }
        self.func
    }

    fn into_inner<'a>(mut self) -> (&'a mut CompilingFunction, Option<Block>)
    where
        Self: 'a,
    {
        self.in_outer();
        (self.func, Some(self.other))
    }
}

impl ListDef<'_> {
    pub fn compute_len(
        mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> Handle<naga::Expression> {
        self.compute_len_inner(ctx, func)
    }
    fn compute_len_inner(
        &mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> Handle<naga::Expression> {
        if let Some(len) = self.cached_len {
            return len;
        }
        let len = match &mut self.inner {
            UntypedListDef::Materialized(materialized_list) => match materialized_list {
                MaterializedList::Temporary(stack_alloc) => {
                    func.compute_stack_list_len(&WithScalarType {
                        ty: self.ty,
                        inner: *stack_alloc,
                    })
                }
                MaterializedList::Static(static_list) => static_list.len,
                MaterializedList::Empty => func.constants.zero_u32,
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
            UntypedListDef::Join(join) => join.compute_len(ctx, func),
            UntypedListDef::Filter(_) => {
                replace_with_or_abort(&mut self.inner, |v| {
                    let UntypedListDef::Filter(f) = v else {
                        unreachable!()
                    };
                    UntypedListDef::Materialized(f.materialize(ctx, func))
                });
                self.compute_len_inner(ctx, func)
            }
            UntypedListDef::Select(select) => {
                let test = *self.cached_select_test.get_or_insert_with(|| {
                    let i = compile_scalar(ctx, func, select.test).inner;
                    func.emit_exprs();
                    i
                });
                let result = func.new_local(ctx.types.u32, None, None);

                let outer = func.new_block();
                let consequent_len = select.consequent_alternate.0.compute_len_inner(ctx, func);
                func.store(result, consequent_len);
                let consequent = func.new_block();

                let alternate_len = select.consequent_alternate.1.compute_len_inner(ctx, func);
                func.store(result, alternate_len);

                let alternate = func.new_block();
                func.body = outer;
                func.body.push(
                    Statement::If {
                        condition: test,
                        accept: consequent,
                        reject: alternate,
                    },
                    naga::Span::UNDEFINED,
                );
                func.load(result)
            }
        };
        self.cached_len = Some(len);
        len
    }
    pub(crate) fn index(
        mut self,
        idx: Handle<naga::Expression>,
        ctx: &mut Compiler,
        blocks: &mut impl MaybeSwitchBlock,
    ) -> Handle<naga::Expression> {
        match self.inner {
            UntypedListDef::Materialized(materialized_list) => {
                materialized_list.index(ctx, blocks.in_inner(), self.ty, idx)
            }
            UntypedListDef::Broadcast(lazy_broadcast) => lazy_broadcast.index(idx, ctx, blocks),
            UntypedListDef::Comprehension(lazy_comprehension) => {
                lazy_comprehension.index(idx, ctx, blocks)
            }
            UntypedListDef::LazyStatic(LazyStaticList { elements }) => {
                //TODO decide when to allocate to stack buffer to avoid overflowing/bloating the GPU stack with massive local arrays
                self.inner = if elements.is_empty() {
                    UntypedListDef::Materialized(MaterializedList::Empty)
                } else {
                    // materialize static lists into the prelude scope
                    let func = blocks.in_outer();
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
                            func.push_frame();
                            let r = compile_scalar(ctx, func, e).inner;
                            func.pop_frame();
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
                self.index(idx, ctx, blocks)
            }
            UntypedListDef::Join(join) => join.index(idx, ctx, blocks),
            UntypedListDef::Filter(filter) => {
                self.inner =
                    UntypedListDef::Materialized(filter.materialize(ctx, blocks.in_outer()));
                self.index(idx, ctx, blocks)
            }
            UntypedListDef::Select(select) => {
                let test = *self.cached_select_test.get_or_insert_with(|| {
                    let i = compile_scalar(ctx, blocks.in_outer(), select.test).inner;
                    blocks.in_outer().emit_exprs();
                    i
                });
                let index_result =
                    blocks
                        .in_outer()
                        .new_local(ctx.scalar_type(self.ty), None, None);

                let func = blocks.in_outer();

                let outer = func.new_block();
                // two blank blocks :3
                let mut blocks_consequent = WithEvalBlock {
                    other: func.new_block(),
                    func,
                    in_eval: false,
                };
                let indexed_consequent =
                    select
                        .consequent_alternate
                        .0
                        .index(idx, ctx, &mut blocks_consequent);
                blocks_consequent
                    .in_inner()
                    .store(index_result, indexed_consequent);
                let WithEvalBlock {
                    other: consequent_outer,
                    func,
                    ..
                } = blocks_consequent;
                let consequent_inner = func.new_block();

                let mut blocks_alternate = WithEvalBlock {
                    other: func.new_block(),
                    func,
                    in_eval: false,
                };
                let indexed_alternate =
                    select
                        .consequent_alternate
                        .1
                        .index(idx, ctx, &mut blocks_alternate);
                blocks_alternate
                    .in_inner()
                    .store(index_result, indexed_alternate);
                let WithEvalBlock {
                    other: alternate_outer,
                    func,
                    ..
                } = blocks_alternate;
                let alternate_inner = func.new_block();
                // back to surrounding block
                func.body = outer;
                // empty block has no side effects by definition
                if !(consequent_outer.is_empty() && alternate_outer.is_empty()) {
                    func.body.push(
                        Statement::If {
                            condition: test,
                            accept: consequent_outer,
                            reject: alternate_outer,
                        },
                        naga::Span::UNDEFINED,
                    );
                }
                blocks.in_inner().body.push(
                    Statement::If {
                        condition: test,
                        accept: consequent_inner,
                        reject: alternate_inner,
                    },
                    naga::Span::UNDEFINED,
                );
                blocks.in_inner().load(index_result)
            }
        }
    }
    pub(crate) fn materialize(
        mut self,
        ctx: &mut Compiler,
        func: &mut CompilingFunction,
    ) -> MaterializedList {
        match self.inner {
            UntypedListDef::Materialized(materialized_list) => materialized_list,
            // filter has a specialized materialize implementation
            UntypedListDef::Filter(filter) => filter.materialize(ctx, func),
            _ => {
                let out_ty = self.ty;
                let len = self.compute_len_inner(ctx, func);
                let iter_index = func.new_local_index(ctx);
                let alloc = func.alloc_list(ctx, self.ty, len);
                let body = func.new_block();

                func.push_frame();
                let this_iter_idx = func.load(iter_index);

                let mut b = WithEvalBlock {
                    other: body,
                    func,
                    in_eval: true,
                };

                let val = self.index(this_iter_idx, ctx, &mut b);
                let func = b.in_inner();

                func.store_index_list_typed(&alloc, this_iter_idx, ScalarRef::new(out_ty, val));
                let next_idx = func.increment(this_iter_idx);

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
                func.pop_frame();
                let _ = b.in_outer();
                let WithEvalBlock {
                    other: body, func, ..
                } = b;
                func.body.push(
                    naga::Statement::Loop {
                        body,
                        continuing: Block::default(),
                        break_if: None,
                    },
                    naga::Span::UNDEFINED,
                );

                MaterializedList::Temporary(alloc.inner)
            }
        }
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
            //TODO when GPU panics are implemented this should be one
            MaterializedList::Empty => {
                func.add_preemit(naga::Expression::ZeroValue(ctx.scalar_type(ty)))
            }
        }
    }
}
pub type StackList = WithScalarType<StackAlloc>;
