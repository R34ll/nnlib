use super::Data;



pub mod sequential_model;
use sequential_model::SequentialModel;




pub trait Model{

    // fn new()->Self; // initilize the model | default implementation

    // fn build(&self,optimizer:Optimizers,loss:Losses)->&Self{
    //     todo!()
    // }


    /// forward across LAYERS one features
    fn forward(&mut self,z:&Data)->Data; // myMOdel::forward(xi)


}








