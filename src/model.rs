use crate::{
    conv::ConvBlock,
};

use burn::{
    module::Module,
    nn, tensor::{backend::Backend, Tensor},
};

use rtt_target::rprintln;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    linear: nn::Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        let conv1 = ConvBlock::new(6, 32, 5, 1);
        let conv2 = ConvBlock::new(32, 32, 5, 3);
        let linear = nn::LinearConfig::new(32, 2).init();

        Model { conv1, conv2, linear }
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.conv1.forward(input);
        let x = self.conv2.forward(x);
        let [batch_size, channel, length] = x.dims();
        let x = x.reshape([length, channel, batch_size]).squeeze(2);
        let x = self.linear.forward(x);
        let x = x.tanh();
        x
    }
}

