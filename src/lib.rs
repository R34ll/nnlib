#![allow(unused)]
extern crate ndarray;
use ndarray::Array2;


/// The network models just accept 2d.
/// Using Ndarray.
pub type Data = Array2<f32>;


pub mod dataset;

/// Module of Activation functionse
pub mod activations; 

/// Module of Neural Network predefineds models
pub mod models; 

/// Module of Optimizers 
pub mod optimizer;

/// Module of Layers
pub mod layer; 

/// Module of Cost functions
pub mod loss; 

pub mod toolkit; 



pub mod model{
	pub use crate::models::{
		Model,
		sequential_model::SequentialModel
	};
}



/// Prelude Module
pub mod prelude{ 

	pub use super::Data;

	pub use super::activations::*;
	pub use super::model::*;
	pub use super::optimizer::*;
	pub use super::layer::*;
	pub use super::loss::*;
	pub use super::toolkit::*;
}







