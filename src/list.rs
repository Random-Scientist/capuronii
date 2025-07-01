use std::collections::HashMap;

use ambavia::type_checker::{Assignment, TypedExpression};
use naga::{Expression, Function, Handle, Statement};

use crate::{BaseType, Compiler, FixedInputId, alloc::StackAlloc};

pub type ScalarSequence<'e> = &'e [TypedExpression];
pub type Join<'e> = Vec<ListDef<'e>>;
#[derive(Debug)]
pub struct RangeList<'e> {
    pub start: &'e TypedExpression,
    pub interval: &'e TypedExpression,
    pub end: &'e TypedExpression,
}
#[derive(Debug)]
pub struct Broadcast<'e> {
    // function(stack_head: ptr<function, u32>, vector input 1, ... vector input over.len()) -> scalar
    pub body: &'e TypedExpression,
    // definitions for Assignment bindings in this scope
    pub scalars: &'e [Assignment],
    pub over: Vec<(usize, ListDef<'e>)>,
}
#[derive(Debug)]
pub struct Comprehension<'e> {
    pub body: &'e TypedExpression,
    // map from assignment to listdef
    pub over: HashMap<usize, ListDef<'e>>,
}
#[derive(Debug)]
pub struct Filter<'e> {
    // invariant Body result is bool
    pub filter: Broadcast<'e>,
    pub source: Box<ListDef<'e>>,
}
#[derive(Debug)]
pub struct FixedMaterialized {
    arr: Handle<Expression>,
    len: u32,
}
#[derive(Debug)]
pub struct Piecewise<'e> {
    // scalar bool
    pub test: &'e TypedExpression,
    pub values: Box<[ListDef<'e>; 2]>,
}
/// Untyped list definition
#[derive(Debug)]
pub(crate) enum ListDef<'e> {
    /// List with fixed 0 length
    Empty,
    /// List consisting of a sequence of scalar values with a fixed length
    Scalars(ScalarSequence<'e>),
    /// List consisting of the result of concatenating a sequence of lists
    Join(Join<'e>),
    /// Range list definition
    Range(RangeList<'e>),
    /// Constant list input
    BufferConstant(FixedInputId),
    /// Broadcast expression
    Broadcast(Broadcast<'e>),
    /// List comprehension expression
    Comprehension(Comprehension<'e>),

    Filter(Filter<'e>),
    Piecewise(Piecewise<'e>),
    /// Dynamically sized list that has been evaluated into a stack buffer
    DynMaterialized(StackAlloc),
    /// Statically sized list that has been evaluated into a const-sized array
    FixedMaterialized(FixedMaterialized),
}
