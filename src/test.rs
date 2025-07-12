use naga::valid::Validator;
use parse::{
    ast_parser::parse_expression_list_entry, latex_parser, name_resolver::resolve_names,
    type_checker::type_check,
};
use typed_index_collections::ti_vec;

use crate::compile;

#[test]
fn test_to_mtl() {
    //let tex = r"c = [ (a, \{ b > 2: [1,2,3,4], a > 2: [4,3,2,1], 1 \}[2] ) \operatorname{for} a = [1,2,3,4], b = [1,3,5] ][2]";
    let tex = r"(c[ 3,2,1 ] \operatorname{with} c = [1,2,3])";
    let tex = latex_parser::parse_latex(tex).unwrap();
    let expr = parse_expression_list_entry(&tex).unwrap();
    let (a, b) = resolve_names(&ti_vec![expr]);
    let (checked, map) = type_check(&a);
    dbg!(&checked);
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
