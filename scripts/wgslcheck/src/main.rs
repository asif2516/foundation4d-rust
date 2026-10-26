fn main() {
    let path = std::env::args().nth(1).expect("shader path");
    let src = std::fs::read_to_string(&path).expect("read shader");
    match naga::front::wgsl::parse_str(&src) {
        Ok(module) => {
            let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default());
            let _ = &mut v;
            match v.validate(&module) {
                Ok(_) => println!("WGSL OK: {} validated", path),
                Err(e) => { println!("VALIDATION ERROR: {e}"); std::process::exit(1); }
            }
        }
        Err(e) => { println!("PARSE ERROR: {e}"); std::process::exit(1); }
    }
}
