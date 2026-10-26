fn main() {
    use std::time::{SystemTime, UNIX_EPOCH};
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => println!("rust epoch ms: {}", d.as_secs_f64()*1000.0),
        Err(e) => println!("ERROR: clock before epoch: {:?}", e),
    }
}
