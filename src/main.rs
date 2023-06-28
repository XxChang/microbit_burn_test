#![allow(clippy::empty_loop)]
#![no_std]
#![no_main]

extern crate alloc;

use panic_rtt_target as _;
use rtt_target::{rtt_init_print, rprintln};

use cortex_m_rt::entry;
use embedded_alloc::Heap;

use microbit as _;

use burn_ndarray::NdArrayBackend;
use burn::tensor::{Distribution::Standard, Tensor};

#[global_allocator]
static HEAP: Heap = Heap::empty();

mod conv;
mod model;

#[entry]
fn main() -> ! {
    rtt_init_print!();

    {
        use core::mem::MaybeUninit;
        const HEAP_SIZE: usize = 10240*4;
        static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];
        unsafe { HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE) }
    }
    
    type Backend = NdArrayBackend<f32>;

    let model = model::Model::<Backend>::new();
    
    let input_shape = [1, 6, 15];

    let input = Tensor::<Backend, 3>::random(input_shape, Standard);

    let output = model.forward(input);

    rprintln!("heap used is {}", HEAP.used()) ;

    for x in output.to_data().value.into_iter() {
        rprintln!("{}", x);
    }

    loop {}
}
