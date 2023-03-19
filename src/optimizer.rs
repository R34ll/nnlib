
use std::fmt::Debug;

use super::prelude::*;
use super::Data;	


use ndarray::Array2;

pub trait OptimAlgorithm:Debug{
	fn optimize(&self,model: &mut Sequential, loss:Data);
}




// stochast gradient descent
#[derive(Debug,Default)]
pub struct SGD{
    pub alpha: f32 // Learning rate
}
impl SGD{
	// set alpha
    pub fn new(&self,alpha:f32)->Self{
        Self { alpha }
    }

	pub fn static_backward(&self,model: &mut Sequential, loss:Data){
		let (w1,b1) = (&model.layers[0].weights,&model.layers[0].bias);
		let  w1_act = &model.layers[0].activation;

		let (w2,b2) = (&model.layers[1].weights,&model.layers[1].bias);
		let w2_act = &model.layers[1].activation;
		
		let l_one_out = &model.grad[0]; // input xi
		let l_two_out = &model.grad[1]; // output layer one
		let output = &model.grad[2];  // output layer two |	(1,1)

		let derive_output = &loss * output;

		let l_one_error = derive_output.dot(&w2.t());
		let derive_w1 = l_one_error * w1_act.act_grad(l_two_out.to_owned()); 
		
		let dw2 = l_two_out.t().dot(&derive_output);
		let dw1 = l_one_out.t().dot(&derive_w1);

		model.layers[0].weights = model.layers[0].weights.clone() - self.alpha * dw1;
		model.layers[1].weights = model.layers[1].weights.clone() - self.alpha * dw2;

		model.layers[0].bias 	= model.layers[0].bias.clone() -  &loss * self.alpha ;
		model.layers[1].bias 	= model.layers[1].bias.clone() -  &loss * self.alpha ;
	}
}

impl OptimAlgorithm for SGD{
	/* PS: I'm still studying about ;-; */
	fn optimize(&self,model: &mut Sequential, loss:Data){
		let n_layers = model.layers.len(); 
		let mut derive = Array2::<f32>::zeros((1,1));

		for idx in (0..n_layers).rev(){

			if idx == n_layers-1{
				let output = &model.grad[idx+1]; //index=2
				let output_l_2 = &&model.grad[idx];

				derive = loss.t().dot(output);
				
				let dw = output_l_2.t().dot(&derive);
				model.layers[idx].weights = model.layers[idx].weights.clone() - self.alpha * dw;
				continue;
			}

			let o  = &model.grad[idx+1];
			let w = &model.layers[idx+1].weights;
			let a = &model.layers[idx].activation;

			derive = derive.dot(&w.t()) * a.act_grad(o.to_owned());
			let dw = &model.grad[idx].t().dot(&derive);


			model.layers[idx].weights = model.layers[idx].weights.clone() - self.alpha * dw;
			model.layers[idx].bias = model.layers[idx].bias.clone() - self.alpha * &loss; 
			


		}

	}
}



pub struct Optimizer;
impl Optimizer{
	pub const SGD:SGD = SGD{alpha:0.01};
}



#[cfg(test)]
mod sequential{
	use super::*;
	
	// #[test]
	fn test_instance(){
		let z = Optimizer::SGD.new(0.01);
			

	}
}






