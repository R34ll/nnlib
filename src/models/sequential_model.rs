use ndarray::Axis;
use crate::{loss::CostFunction, optimizer::OptimAlgorithm};
use crate::prelude::*;


#[derive(Debug)]
pub struct SequentialModel{
    pub layers:Sequential,
    loss:Box<dyn CostFunction>,
    optmizer:Box<dyn OptimAlgorithm>,
    // loss_history:Option<Vec<f32>>
}

impl Default for SequentialModel{
    fn default() -> Self {
        Self { 
            layers: Sequential::new(), 
            loss: Box::new(MeanSquaredError), 
            optmizer: Box::new(SGD{alpha:0.01}) 
        }
    }
}


impl SequentialModel {
    pub fn new()->Self{
        SequentialModel::default()        
    }

    pub fn layer<A:ActFunc+'static>(&mut self,input:usize,output:usize,activation:A)->&mut Self{
        let z = self.layers.add(input, output, activation);
        self
    }

    pub fn build<L:CostFunction+'static, O:OptimAlgorithm+'static>(&mut self,loss:L, optimizer:O)->&mut Self{
        self.loss = Box::new(loss);
        self.optmizer = Box::new(optimizer);
        self
    }

    pub fn train(&mut self,x_train: &Data, y_train: &Data, epochs:usize) -> Vec<f32>{

        let mut hist =Vec::<f32>::new();

// 1. shuffle the dataset

        for _epoch in 0..epochs{
            let mut loss_hist = 0.0;
            for idx in 0..x_train.shape()[0]{
                let xi:Data = x_train.index_axis(Axis(0), idx).insert_axis(Axis(0)).to_owned(); 
                let yi:Data = y_train.index_axis(Axis(0), idx).insert_axis(Axis(0)).to_owned(); 

                let output = self.forward(&xi);

                let loss = self.loss.cost_grad(output.clone(), yi.clone());
                self.optmizer.optimize(&mut self.layers, loss);

                loss_hist += self.loss.cost(output,yi);
            }
            hist.push(loss_hist);
            println!("{} - {}",_epoch,loss_hist);
        } 
        hist
    }
    
}

impl Model for SequentialModel{
    fn forward(&mut self,z:&Data)-> Data {
        self.layers.forward(z).to_owned()
    }
}



impl std::fmt::Display for SequentialModel{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut text = String::new();
        text.push_str(&format!("Loss={:?}\n Optimizer={:?}\n",self.loss,self.optmizer));

        text.push_str(" Layers: \n");
        for (idx,layer) in self.layers.layers.iter().enumerate(){
            let (inp,out) = (layer.weights.shape()[0], layer.weights.shape()[1]);

            let t = format!(" {} | shape=({},{}), Activation={:?}\n",
                idx,
                inp,
                out,
                layer.activation,
            );
            
            text.push_str(&t);
        }

        write!(f," {}", text)
    }
}









