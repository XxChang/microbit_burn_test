use burn::{
    module::Module,
    nn, tensor::{backend::Backend, activation, Tensor},
};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: nn::conv::Conv1d<B>,
    activation: nn::ReLU,
    dropout: nn::Dropout,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels_in: usize, channels_out: usize, kernel_size: usize, dilation: usize) -> Self {
        let conv1 = nn::conv::Conv1dConfig::new(channels_in, channels_out, kernel_size).with_dilation(dilation)
            .with_padding(nn::conv::Conv1dPaddingConfig::Same).init();
        let activation = nn::ReLU::new();
        let dropout = nn::DropoutConfig::new(0.5).init();
        
        ConvBlock { conv1, activation, dropout }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        x
    }
}