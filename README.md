# nnlib
An experimental machine learning framework written in rust. 

That project as developed to deepen my knowledge of machine learning and rust.


## Example of usage
```rust
use nnlib::prelude::*;

fn main() {

    let (x_train, 
        y_train, 
        x_test, 
        y_test) = get_dataset(Dataset::Iris);

    let epochs = 15;
    let lr = 0.01;

    let mut model = SequentialModel::new();

    model.layer(4, 5, Activation::Linear);
    model.layer(5, 1, Activation::Sigmoid);

    model.build(Loss::MeanSquaredError, Optimizer::SGD.new(lr));
  
    model.train(&x_train, &y_train, epochs);
}
```
