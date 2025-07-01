use ambavia::{
    ast_parser::{parse_expression_list_entry, parse_nodes_into_expression},
    latex_parser,
    latex_tree_flattener::flatten,
    name_resolver::{ExpressionIndex, resolve_names},
    type_checker::type_check,
};
use naga::valid::{ModuleInfo, Validator};
use typed_index_collections::ti_vec;

use crate::compile;

#[test]
fn test_to_mtl() {
    let tex = r"a = [1,2,3,4, 2+ 2][1+2]";
    let tex = latex_parser::parse_latex(tex).unwrap();
    let expr = parse_expression_list_entry(&tex).unwrap();
    let (a, b) = resolve_names(&ti_vec![expr]);
    let (checked, map) = type_check(&a);
    let expr = &checked.first().unwrap().value;
    let m = compile(expr);
    let mut v = Validator::new(Default::default(), Default::default());
    dbg!(&m);
    let info = v.validate(&m).unwrap();
    println!(
        "Compiled MSL: {}",
        naga::back::msl::write_string(&m, &info, &Default::default(), &Default::default(),)
            .unwrap()
            .0
    );
    let mut s = String::new();
    let c = Default::default();
    let mut w = naga::back::hlsl::Writer::new(&mut s, &c);
    w.write(&m, &info, None).unwrap();
    println!("Compiled HLSL: {}", s);
}
