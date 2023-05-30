use ndarray::Array2;
use rand::{distributions::Uniform, Rng}; 
use crate::toolkit::xavier_initialization;

use super::Data;
use super::activations::{ActFunc, *};
use super::toolkit::build_random_weight;

/// Linear Layer
#[derive(Debug)]
pub struct Layer{
    pub weights:Data,
    pub bias:Data,
    pub activation:Box<dyn ActFunc>,
}

impl Layer{

    pub fn new<T:ActFunc+'static>(input:usize,output:usize,activation:T)->Self{
		// let weights =  Array2::<f32>::ones((input,output));
		// let bias = Array2::<f32>::ones((1,output));


		// let weights =  Array2::<f32>::zeros((input,output));
		// let bias = Array2::<f32>::zeros((1,output));


		// let weights = build_random_weight(input,output); // rand 
		// let bias = build_random_weight(1,output); // rand


		let uniform = false;
		let weights = xavier_initialization(input,output,uniform);
		let bias = xavier_initialization(1,output,uniform);




		Self { weights, bias,activation: Box::new(activation) }
	}

	pub fn forward(&self,z:&Data)->Data{
		let linear = z.dot(&self.weights) + &self.bias;
		self.activation.call(linear)
	}


}

#[derive(Debug,Default)]
pub struct Sequential{
	pub layers:Vec<Layer>,
	pub size:usize,
	pub grad:Vec<Data>,
}
impl Sequential{
	
	
	pub fn new()->Self{Self { 
		..Default::default()
	}}

	pub fn add<Act:ActFunc+'static>(&mut self,input:usize,output:usize,activation:Act)->&mut Self{ // amke return &self 
		let mut l = Layer::new(input, output, activation);
		self.layers.push(l);
		self.size+=1;
		self
	}

	pub fn forward(&mut self,mut z: &Data)->Data{ // make accept reference
		self.grad.push(z.clone());

		let mut output = z.to_owned();

		for layer in self.layers.iter(){ // use map
			output = layer.forward(&output);
			self.grad.push(output.clone());
		}

		output
	}
}
















///////////////////////////////////////////////////////////// test

/// Sequential Layer
#[derive(Debug,Default)]
pub struct SequentialBuilder{
	pub layers:Vec<Layer>,
}

impl SequentialBuilder
{
	pub fn new()->Self{Self { 
		..Default::default()
	}}
	pub fn add<Act:ActFunc+'static>(&mut self,input:usize,output:usize,activation:Act)->&mut Self{ // amke return &self 
		let mut l = Layer::new(input, output, activation);
		self.layers.push(l);

		
		self
	}
	// build a sequential model from a vector
	pub fn from_vec(vec:Vec<Layer>)->Self{
		Self{layers:vec}
	}


	pub fn build(self)->Sequential{
		Sequential{
			size:self.layers.len(),
			layers:self.layers,
			..Default::default()
		}
	}


}



struct TestLayerBuilder{
	weight:Data,
	bias:Option<Data>,
	activation:Box<dyn ActFunc>,
	shape:(usize,usize),
}
impl TestLayerBuilder{
	pub fn new()->Self{
		Self{
			weight:Array2::<f32>::ones((1,1)),
			bias:Some(Array2::<f32>::ones((1,1))), // Optino<Data>
			activation: Box::new(Linear),
			shape:(1,1),
		}
	}
	pub fn activation<Act:ActFunc+'static>(mut self,activation:Act)->Self{
		self.activation = Box::new(activation);
		self
	}
	pub fn bias(mut self,bias:bool)->Self{
		if !bias{self.bias = None}
		self
	}
	pub fn zeros(mut self,shape:(usize,usize))->Self{
		self.weight = Array2::<f32>::zeros((shape.0,shape.1));
		if self.bias.is_some(){
			self.bias = Some(Array2::<f32>::zeros((1,shape.1)));
		}
		self
	}
	pub fn ones(mut self,shape:(usize,usize))->Self{
		self.weight = Array2::<f32>::ones((shape.0,shape.1));
		if self.bias.is_some(){
			self.bias = Some(Array2::<f32>::ones((1,shape.1)));
		}
		self
	}
	pub fn uniform(mut self, from:f32,to:f32,shape:(usize,usize))->Self{
		let mut rng = rand::thread_rng();
		let range = Uniform::new(from, to);
		let vals_w: Vec<f32> = (0..shape.0*shape.1).map(|_| rng.sample(range)).collect();
		self.weight = Array2::from_shape_vec((shape.0,shape.1), vals_w).unwrap();
		self.shape = shape;
		if self.bias.is_some(){
			let vals_b: Vec<f32> = (0..shape.1).map(|_| rng.sample(range)).collect();
			self.bias = Some(Array2::from_shape_vec((1,shape.1), vals_b).unwrap());
		}
		self
	}

	pub fn build(self)->TestLayer{
		TestLayer { 
			weight: self.weight, 
			bias: self.bias, 
			activation:self.activation, 
			shape: self.shape, 
			output: Array2::<f32>::zeros((1,1))
		 }
	}



}

#[derive(Debug)]
pub struct TestLayer{
	weight:Data,
	bias:Option<Data>,
	activation:Box<dyn ActFunc>,
	shape:(usize,usize),
	output:Data
}
impl TestLayer{
	pub fn forward(mut self,z:Data)->Data{
		let lin = {
			if self.bias.is_some(){z.dot(&self.weight) + self.bias.unwrap()}
			else{z.dot(&self.weight)}
		};
		self.output = self.activation.call(lin);
		self.output
	}
}








