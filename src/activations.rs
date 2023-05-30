
use super::Data;
use super::toolkit::{max,exp};
use ndarray::{Axis, Array2, stack};


/// Trait for activation functions models.
// Activation fucntions trait
pub trait ActFunc:std::fmt::Debug{
    /// Activation function
    fn call(&self,z:Data)->Data;

    /// The gradient/derive of the activation function
    fn act_grad(&self,z:Data)->Data;
}

/// The Sigmoid activation function
#[derive(Debug,Default,Clone)]
pub struct Sigmoid;
impl ActFunc for Sigmoid{

    // Sigmoid function.
    fn call(&self,z:Data)->Data {
	    z.mapv(|x|{
            1. / (1. + std::f32::consts::E.powf(-x))
	    })   
    }

    /// derive of sigmoid function
    fn act_grad(&self,z:Data)->Data {z.clone() * (1.0 - z)}
}

/// The Linear activation function.
#[derive(Debug,Default,Clone)]
pub struct Linear;// None
impl ActFunc for Linear{
    fn call(&self,z:Data)    ->Data {z}
    fn act_grad(&self,z:Data)->Data {z}
}


/// The ReLU activation function.
#[derive(Debug,Default,Clone)]
pub struct ReLU;
impl ActFunc for ReLU{
    fn call(&self,z:Data)->Data {
        z.mapv(|x|{if x < 0.0{0.0}else{x}})
    }
    fn act_grad(&self,z:Data)->Data {
        z.mapv(|x| {if x < 0.0{0.0}else{1.0}})
    }
}


/// The Leaky ReLU activation function
#[derive(Debug, Default, Clone)]
pub struct LeakyReLU;
impl ActFunc for LeakyReLU{
    fn call(&self, z:Data)->Data{
        z.mapv(|x|{if x < 0.0 { 0.01*x }else{ x } })
    }

    fn act_grad(&self, z:Data)->Data{
        z.mapv(|x|{if x < 0.0 { 0.01*x }else{ x }})
    }
}


/// The Softmax activation function.
#[derive(Debug,Default,Clone)]
pub struct Softmax;
impl ActFunc for Softmax{
    fn call(&self,z:Data)->Data {
        // let e_x = exp(z.clone() - max(z));
        // e_x.clone() / e_x.sum_axis(Axis(0)) 
    
        let e_x = exp(z);
        e_x.clone()/e_x.sum()


    }

    fn act_grad(&self,z:Data)->Data {
        z.clone() - (1.0 - z) // not sure if is correct. test with a cost function
    }
}


// /// Activations functions collect
// #[derive(Debug,Clone)]
// pub struct Activation; 
// impl Activation{

//     #[allow(non_upper_case_globals)]
// 	pub const Sigmoid:Sigmoid = Sigmoid;

//     #[allow(non_upper_case_globals)]
//     pub const Linear:Linear = Linear;

//     #[allow(non_upper_case_globals)]
// 	pub const ReLU:ReLU = ReLU;

//     #[allow(non_upper_case_globals)]
//     pub const Softmax:Softmax = Softmax;
// }



#[cfg(test)]
mod test_activation{
    use ndarray::arr2;
    use super::{*,ActFunc};

    // Using Data = Array2<f32>
    const DATA: [[f32; 3]; 2] = [[4.0,-5.0,-6.0],[-7.0,8.0,9.0]];

    #[test]
    // #[ignore = "Still developing"] 
    fn softmax(){
        let softmax = Softmax;
        let output = softmax.call(arr2(&DATA));

        assert_eq!(
            output,
            arr2(& [[0.0049016858, 6.0491624e-7, 2.2253624e-7],
                [8.186651e-8, 0.26762295, 0.7274745]]))
    }




    #[test]
    fn sigmoid(){
        let sigmoid = Sigmoid;
        let output = sigmoid.call(arr2(&DATA));

        assert_eq!(
            output,
            arr2(&[[0.98201376, 0.0066928524, 0.0024726237],
                [0.0009110514, 0.99966466, 0.9998766]])
        )
    }

    #[test]
    fn linear(){
        let linear = Linear;
        let output = linear.call(arr2(&DATA));

        assert_eq!(
            output,
            arr2(&DATA)
        )
    }

    #[test]
    fn relu(){
        let relu = ReLU;
        let output = relu.call(arr2(&DATA));

        assert_eq!(
            output,
            arr2(&[[4.0, 0.0, 0.0],
                [0.0, 8.0, 9.0]])
        )
    }










}






