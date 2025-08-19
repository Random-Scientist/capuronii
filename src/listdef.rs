use std::num::NonZeroU32;

use naga::{Block, Handle, Statement};
use parse::type_checker::{self, Assignment, BaseType};
use replace_with::replace_with_or_abort;

use crate::{
    ArenaExt, Compiler, ScalarValue,
    alloc::{Alloc, StackAlloc, StaticAlloc},
    compile_scalar,
    function::CompilingFunction,
    math_impl::Float32,
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
    ) -> ScalarValue {
        let p = func.in_outer();
        p.bind_assignments(ctx, self.scalars);

        // bind varyings
        for (id, list) in self.varying {
            let assignment = func.in_inner().get_scalar_assignment(ctx, id, list.ty);
            let value = list.index_direct(func, ctx, idx);
            let inner = func.in_inner();
            let s = inner.serialize_scalar(ctx, value);
            inner.store(assignment, s);
        }
        let eval_in = func.in_inner();
        eval_in.push_frame();
        let r = compile_scalar(ctx, eval_in, self.body);
        eval_in.pop_frame();
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
    ) -> ScalarValue {
        let pre = blocks.in_outer();
        let lengths: Vec<_> = self
            .varying
            .iter_mut()
            .map(|(_, l)| l.compute_len_inner(ctx, pre))
            .collect();
        // Determine and index comprehension varyings
        // TODO make this only update a given varying binding when its index changes
        let mut prev_section_len = pre.consts.one_u32;
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
            let val = value.index_direct(blocks, ctx, index);

            let func = blocks.in_inner();
            let s = func.get_scalar_assignment(ctx, assignment, out_ty);
            let ser = func.serialize_scalar(ctx, val);
            func.store(s, ser);
        }
        let func = blocks.in_inner();

        // frame for listcomp deps

        func.push_frame();
        func.bind_assignments(ctx, &self.body.assignments);
        let result = compile_scalar(ctx, func, &self.body.value);
        func.pop_frame();

        result
    }
}

pub struct LazyStaticList<'a> {
    pub(crate) elements: &'a [type_checker::TypedExpression],
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
    ) -> ScalarValue {
        let pre = blocks.in_outer();
        let mut curr_start_idx = pre.consts.zero_u32;

        let val_out = pre.new_local(
            ctx.scalar_type_repr(self.lists.first().unwrap().ty),
            None,
            Some(format!("join_output_{}", pre.func.local_variables.len())),
        );

        let mut body_block = blocks.in_inner().new_block();

        let mut cond_blocks = Vec::with_capacity(self.lists.len());
        let mut ty = None;
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

            let val = list.index_direct(blocks, ctx, transformed_idx);
            ty = Some(val.ty());
            let func = blocks.in_inner();

            let ser = func.serialize_scalar(ctx, val);
            func.store(val_out, ser);

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
        func.func
            .body
            .push(naga::Statement::Block(nested_if), naga::Span::UNDEFINED);

        let l = func.load(val_out);
        func.deserialize_scalar(ctx, l, ty.unwrap())
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
        let result = self.src.index_direct(&mut other, ctx, this_iter_index);

        let func = other.in_inner();

        func.store_to_top_of_stack_typed(ctx, result);

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
        func.push_statement(naga::Statement::If {
            condition: test.bool(),
            accept: success_case,
            reject: Block::new(),
        });
        let next_iter_index = func.increment(this_iter_index);

        let should_break = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Greater,
            left: next_iter_index,
            right: max_len,
        });
        func.emit_exprs();
        func.push_statement(naga::Statement::If {
            condition: should_break,
            accept: Block::from_vec(vec![naga::Statement::Break]),
            reject: Block::new(),
        });
        func.store(iter_index, next_iter_index);
        // ensure func contains the primary body block, not the loop body block
        let _ = other.in_outer();
        let WithEvalBlock {
            other: body, func, ..
        } = other;

        func.push_statement(naga::Statement::Loop {
            body,
            continuing: Block::new(),
            break_if: None,
        });
        let len = func.load(out_len);

        MaterializedList::Allocated(
            StackAlloc {
                base_addr: out_base_addr,
                size: len,
            }
            .into(),
        )
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
    Allocated(Alloc),
    Empty,
}

pub(crate) trait MaybeSwitchBlock: Sized {
    fn in_outer(&mut self) -> &mut CompilingFunction;
    fn in_inner(&mut self) -> &mut CompilingFunction {
        // default implementation does eval and prelude in the same block
        self.in_outer()
    }
}
impl MaybeSwitchBlock for &'_ mut CompilingFunction {
    fn in_outer(&mut self) -> &mut CompilingFunction {
        self
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
                MaterializedList::Allocated(alloc) => func.len_from_size(*alloc, self.ty),
                MaterializedList::Empty => func.consts.zero_u32,
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
                    let i = compile_scalar(ctx, func, select.test).bool();
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
                func.func.body = outer;
                func.push_statement(Statement::If {
                    condition: test,
                    accept: consequent,
                    reject: alternate,
                });
                func.load(result)
            }
        };
        self.cached_len = Some(len);
        len
    }
    pub(crate) fn index(
        mut self,
        mut func: &mut CompilingFunction,
        ctx: &mut Compiler,
        index: Float32,
    ) -> ScalarValue {
        let len = self.compute_len_inner(ctx, func);
        let one = Float32::new_assume_finite(func.consts.one_f32);
        // additionally serves as a NaN check, since NaN > n will return false
        let check = func.cmp(
            ctx,
            type_checker::ComparisonOperator::GreaterEqual,
            index,
            one,
        );

        let idx_p_1 = func.add_unspanned(naga::Expression::As {
            expr: index.value,
            kind: naga::ScalarKind::Uint,
            convert: Some(4),
        });
        let idx = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Subtract,
            left: idx_p_1,
            right: func.consts.one_u32,
        });

        let boundck = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::Less,
            left: idx,
            right: len,
        });
        let check = func.add_unspanned(naga::Expression::Binary {
            op: naga::BinaryOperator::LogicalAnd,
            left: check,
            right: boundck,
        });
        let nan = func.nan_value_for_ty(ctx, self.ty);
        let indexed = self.index_direct(&mut func, ctx, idx);
        func.select_scalar(ctx, check, indexed, nan)
    }
    pub(crate) fn index_direct(
        mut self,
        blocks: &mut impl MaybeSwitchBlock,
        ctx: &mut Compiler,
        idx: Handle<naga::Expression>,
    ) -> ScalarValue {
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
                    let size =
                        elements.len() as u32 * blocks.in_outer().comptime_size_of_scalar(self.ty);
                    // materialize static lists into the prelude scope
                    let func = blocks.in_outer();
                    let new_arr_ty = ctx.module.types.add_unspanned(naga::Type {
                        name: None,
                        inner: naga::TypeInner::Array {
                            base: ctx.types.u32,
                            size: naga::ArraySize::Constant(NonZeroU32::new(size).unwrap()),
                            stride: 4,
                        },
                    });
                    let components = elements
                        .iter()
                        .flat_map(|e| {
                            func.push_frame();
                            let r = compile_scalar(ctx, func, e);
                            func.pop_frame();
                            func.get_boxed_bits(ctx, r)
                        })
                        .collect();
                    let arr = func.add_unspanned(naga::Expression::Compose {
                        ty: new_arr_ty,
                        components,
                    });
                    let loc = func.new_local(new_arr_ty, None, None);
                    func.store(loc, arr);

                    UntypedListDef::Materialized(MaterializedList::Allocated(Alloc::Static(
                        StaticAlloc {
                            array_pointer: loc,
                            size,
                        },
                    )))
                };
                self.index_direct(blocks, ctx, idx)
            }
            UntypedListDef::Join(join) => join.index(idx, ctx, blocks),
            UntypedListDef::Filter(filter) => {
                self.inner =
                    UntypedListDef::Materialized(filter.materialize(ctx, blocks.in_outer()));
                self.index_direct(blocks, ctx, idx)
            }
            UntypedListDef::Select(select) => {
                let test = *self.cached_select_test.get_or_insert_with(|| {
                    let out = blocks.in_outer();
                    let i = compile_scalar(ctx, out, select.test);
                    out.emit_exprs();
                    i.bool()
                });
                let index_result =
                    blocks
                        .in_outer()
                        .new_local(ctx.scalar_type_repr(self.ty), None, None);

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
                        .index_direct(&mut blocks_consequent, ctx, idx);
                let ser = blocks_consequent
                    .in_inner()
                    .serialize_scalar(ctx, indexed_consequent);
                blocks_consequent.in_inner().store(index_result, ser);
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
                        .index_direct(&mut blocks_alternate, ctx, idx);
                let ser = blocks_alternate
                    .in_inner()
                    .serialize_scalar(ctx, indexed_alternate);
                blocks_alternate.in_inner().store(index_result, ser);
                let WithEvalBlock {
                    other: alternate_outer,
                    func,
                    ..
                } = blocks_alternate;
                let alternate_inner = func.new_block();
                // back to surrounding block
                func.func.body = outer;
                // empty block has no side effects by definition
                if !(consequent_outer.is_empty() && alternate_outer.is_empty()) {
                    func.push_statement(Statement::If {
                        condition: test,
                        accept: consequent_outer,
                        reject: alternate_outer,
                    });
                }
                blocks.in_inner().push_statement(Statement::If {
                    condition: test,
                    accept: consequent_inner,
                    reject: alternate_inner,
                });
                let inner = blocks.in_inner();
                let l = inner.load(index_result);
                inner.deserialize_scalar(ctx, l, self.ty)
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
                let iter_ofs = func.new_local_index(ctx);
                let ss = func.size_of_scalar(out_ty);
                let alloc_size = func.add_preemit(naga::Expression::Binary {
                    op: naga::BinaryOperator::Multiply,
                    left: len,
                    right: ss,
                });
                let alloc = func.alloc_stack(ctx, alloc_size);

                let body = func.new_block();

                func.push_frame();
                let this_iter_ofs = func.load(iter_ofs);

                let mut b = WithEvalBlock {
                    other: body,
                    func,
                    in_eval: true,
                };

                let val = self.index_direct(&mut b, ctx, this_iter_ofs);
                let func = b.in_inner();

                func.store_to_alloc_typed(ctx, alloc, this_iter_ofs, val);
                let next_ofs = func.add_unspanned(naga::Expression::Binary {
                    op: naga::BinaryOperator::Add,
                    left: this_iter_ofs,
                    right: ss,
                });

                let should_break = func.add_unspanned(naga::Expression::Binary {
                    op: naga::BinaryOperator::GreaterEqual,
                    left: next_ofs,
                    right: alloc_size,
                });
                // barrier
                func.emit_exprs();

                func.push_statement(Statement::If {
                    condition: should_break,
                    accept: Block::from_vec(vec![Statement::Break]),
                    reject: Block::new(),
                });
                func.pop_frame();
                let _ = b.in_outer();
                let WithEvalBlock {
                    other: body, func, ..
                } = b;
                func.push_statement(naga::Statement::Loop {
                    body,
                    continuing: Block::default(),
                    break_if: None,
                });

                MaterializedList::Allocated(alloc)
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
    ) -> ScalarValue {
        match self {
            MaterializedList::Allocated(stack_alloc) => {
                let offset = func.index_to_raw(index, ty);
                func.load_from_alloc_typed(ctx, *stack_alloc, offset, ty)
            }

            //TODO when GPU panics are implemented this should be one
            MaterializedList::Empty => func.zeroed(ty),
        }
    }
}
