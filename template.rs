#![allow(unused_imports, dead_code, non_camel_case_types)]

// use num::integer::{gcd,lcm};
use std::collections::*;
use std::cmp::{max,min};
use crate::scanner::*;

type int = i64;
type float = f64;

fn main(){
    let (r, w) = (std::io::stdin(), std::io::stdout());
    let mut sc = IO::new(r.lock(), w.lock());
    
}

pub mod scanner {
    pub struct IO<R, W: std::io::Write>(R, std::io::BufWriter<W>);

    impl<R: std::io::Read, W: std::io::Write> IO<R, W> {
        pub fn new(r: R, w: W) -> Self {
            Self(r, std::io::BufWriter::new(w))
        }
        pub fn write<S: ToString>(&mut self, s: S) {
            use std::io::Write;
            self.1.write_all(s.to_string().as_bytes()).unwrap();
        }
        pub fn writeline<S: ToString>(&mut self, s: S) {
            use std::io::Write;
            self.1.write_all((s.to_string()+"\n").as_bytes()).unwrap();
        }
        pub fn read<T: std::str::FromStr>(&mut self) -> T {
            use std::io::Read;
            let buf = self
                .0
                .by_ref()
                .bytes()
                .map(|b| b.unwrap())
                .skip_while(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
                .take_while(|&b| b != b' ' && b != b'\n' && b != b'\r' && b != b'\t')
                .collect::<Vec<_>>();
            unsafe { std::str::from_utf8_unchecked(&buf) }
                .parse()
                .ok()
                .expect("Parse error.")
        }
        pub fn readline<T: std::str::FromStr>(&mut self) -> Vec<T> {
            use std::io::Read;
            let buf = self
                .0
                .by_ref()
                .bytes()
                .map(|b| b.unwrap())
                .skip_while(|&b| b == b' ' || b == b'\n' || b == b'\r' || b == b'\t')
                .take_while(|&b| b != b'\n')
                .collect::<Vec<_>>();
            unsafe { std::str::from_utf8_unchecked(&buf) }
                .parse::<String>()
                .ok()
                .expect("Parse error.")
                .split_whitespace()
                .map(|f| f.parse::<T>().ok().unwrap())
                .collect::<Vec<T>>()
        }
        pub fn vec<T: std::str::FromStr>(&mut self, n: usize) -> Vec<T> {
            (0..n).map(|_| self.read()).collect()
        }
        pub fn chars(&mut self) -> Vec<char> {
            self.read::<String>().chars().collect()
        }
    }
}
