
use super::{Data, activations::{Softmax, ActFunc}, toolkit::log};
extern crate ndarray;
use ndarray::{arr2, Axis};



//////////////////////////////////////////////////
/// Calc the delta
pub trait CostFunction:std::fmt::Debug{

    /// Cost function.
    /// Evaluate the cost function
    fn cost(&self,y_hat:Data, yi:Data)->f32;

    /// Gradient of cost function
    fn cost_grad(&self,y_hat:Data, yi:Data)->Data;
}


/// Mean Squared Error
#[derive(Debug,Default)]
pub struct MeanSquaredError;
impl CostFunction for MeanSquaredError{
    fn cost(&self,y_hat:Data, yi:Data)->f32 {
        (y_hat - yi).mapv(|a|a.powf(2.0)).mean().unwrap()
    }
    fn cost_grad(&self,y_hat:Data, yi:Data)->Data {
        y_hat - yi
    }
}

#[derive(Debug,Default)]
pub struct CrossEntropy;
impl CostFunction for CrossEntropy{
    fn cost(&self,y_hat:Data, yi:Data)->f32 {
        let sf = Softmax::default();
        let n = y_hat.shape()[0];
        let mut loss = arr2(&[[0.0]]);

        for i in 0..n{
            let xi = y_hat.index_axis(Axis(0), i).to_owned().insert_axis(Axis(1));
            let out = sf.call(xi);
            let targ = yi.index_axis(Axis(0), i).to_owned().insert_axis(Axis(1));
            loss = loss.clone() + targ * -log(out) / n as f32;
        }
        loss.sum()
    }

    fn cost_grad(&self,y_hat:Data, yi:Data)->Data {
        y_hat - yi
    }
}

#[derive(Debug)]
pub struct Loss;
impl Loss{
    #[allow(non_upper_case_globals)]
    pub const MeanSquaredError:MeanSquaredError = MeanSquaredError; // make accept reference &

    #[allow(non_upper_case_globals)]
    pub const CrossEntropy:CrossEntropy = CrossEntropy;
}
