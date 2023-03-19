use super::Data;
use ndarray::{s, Array2, ArrayBase,Ix2, Data as NdData};
use rand::{distributions::Uniform, Rng,thread_rng}; 
use std::f32::consts::E;


pub fn log(z:Data)->Data{
    z.mapv(|a|a.log(E))
}

pub fn exp(z:Data)->Data{
    z.mapv(|a|E.powf(a)  )
}

pub fn max(z:Data)->f32{
    z.iter().clone().fold(0f32,|var,new| {f32::max(var,*new)})
}

pub fn min(z:Data) ->f32{
    z.iter().clone().fold(0f32,|var,new| {f32::min(var,*new)})
}

pub fn sqrt(z:Data)->Data{
    z.mapv(|a| a.sqrt())
}


pub fn get_batch(dataset:Data,size:usize)->Vec<Data>{

    let n_samples = dataset.shape()[0];
    let batch_size = n_samples/size; // checked_div . 

    if n_samples as f32 %  (n_samples as f32/size as f32) != 0.0{panic!("Batch size don't can be set")}
    let mut new_dataset:Vec<Data> = Vec::new();

    let mut cp = 0;
    for _ in 0..size{ ///////////////////////////////// use .cycle()
        let new = dataset.slice(s![cp..cp+batch_size,..]);
        cp +=batch_size;
        new_dataset.push(new.to_owned());
    }
    new_dataset
}


pub fn build_random_weight(n_inputs:usize,n_outputs:usize)->Data{
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-05.0, 05.0);
    
    let vals: Vec<f32> = (0..n_inputs*n_outputs).map(|_| rng.sample(uniform)).collect();
    Array2::from_shape_vec((n_inputs, n_outputs), vals).unwrap()
}


//////////////////////////////


pub fn xavier_initialization(n_inputs: usize, n_outputs: usize, uniform: bool) ->Data {

    if uniform {
        let mut rng = thread_rng();
        let init_range = (6.0 / (n_inputs + n_outputs) as f64).sqrt() as f32;
        let uniform = Uniform::new(-init_range, init_range);
        
        let vals: Vec<f32> = (0..n_inputs*n_outputs).map(|_| rng.sample(uniform)).collect();
        Array2::from_shape_vec((n_inputs, n_outputs), vals).unwrap()

    } else { // recommended of use

        let stddev = (3.0 / (n_inputs + n_outputs) as f64).sqrt() as f32;
        let normal = thread_rng().gen_range(-stddev..stddev);


        let vals: Vec<f32> = (0..n_inputs*n_outputs).map(|_| normal).collect();
        Array2::from_shape_vec((n_inputs, n_outputs), vals).unwrap()
    }
}



#[cfg(test)]
mod test_toolkit{
    use ndarray::arr2;
    use super::*;

    #[test]
    fn log_(){
        let a = arr2(&[[5.0]]);
        let b = arr2(&[[45.0]]);
        let c = arr2(&[[65.9]]);

        let a_out = log(a);
        let b_out = log(b);
        let c_out = log(c);

        assert_eq!(a_out,arr2(&[[1.6094381]]));
        assert_eq!(b_out,arr2(&[[3.8066628]]));
        assert_eq!(c_out,arr2(&[[4.188139]]));

    }

}


