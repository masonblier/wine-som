extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;
extern crate rusticsom;

use csv::{ReaderBuilder};
use ndarray::prelude::*;
use ndarray_csv::{Array2Reader};
use rusticsom::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::env;

/// number of iterations to run the som trainer
const TRAINING_ITERATIONS: u32 = 1000;
// width and height of the self-organizing map
const SOM_WIDTH: usize = 10;
const SOM_HEIGHT: usize = 10;
// depth of the som, tbh i'm not sure what this does
const SOM_DEPTH: usize = 4;

/// application entry point
fn main() -> Result<(), Box<dyn Error>> {
    // get input file path from command line argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: `cargo run path/to/winequality-red.csv`");
        println!("  file must be specified");
        return Ok(());
    }
    let filename = &args[1];

    // read input csv into ndarray
    let file = File::open(filename)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut all_data: Array2<f64> = reader.deserialize_array2_dynamic()?;

    // slice off 'quality' column
    let data = &mut all_data.slice_mut(s![0.., 0..-1]);
    let nrows = data.nrows();
    let ncols = data.ncols();
    
    // normalize - get sums for each axis and scale values into 0.0-1.0 range
    let col_max = data.fold_axis(Axis(0), f64::MIN, |m, i| (*m).max(*i));
    for r in 0..nrows {
        let mut row = data.row_mut(r);
        for c in 0..ncols {
            row[c] = row[c] / col_max[c];
        }
    }

    // train self-organizing map
    let processed_data = data.to_owned();
    let mut som = SOM::create(SOM_WIDTH, SOM_HEIGHT, SOM_DEPTH, false, None, None, None, None);
    som.train_random(processed_data.clone(), TRAINING_ITERATIONS);

    // get collection 'quality' values from all winner rows for each position in SOM 
    let mut mean_map: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    for x in 0..nrows {
        let y = processed_data.row(x).to_owned();
        let tup = som.winner(y);
        if !mean_map.contains_key(&tup) {
            mean_map.insert(tup, Vec::<u32>::new());
        }
        mean_map.get_mut(&tup).unwrap().push(all_data.row(x)[11].round() as u32);
    }

    // find the mode of each entry in the collection of 'quality' values
    let mut mode_map = Array2::<u32>::zeros((SOM_WIDTH, SOM_HEIGHT));
    for w in 0..SOM_WIDTH {
        let mut row = mode_map.row_mut(w);
        for h in 0..SOM_HEIGHT {
            row[h] = mcv(mean_map.get(&(w,h)).unwrap());           
        }
    }


    // prints the distance value for each cell in som
    let dist_map = som.distance_map();
    println!("Distance Map:");
    println!("{}", dist_map);
    println!();
    // prints the most frequent quality value related to som cell
    println!("Most frequent quality value:");
    println!("{}, ", mode_map);

    // end
    Ok(())
}   

/// helper function to find the most common value of a vec
fn mcv(numbers: &Vec<u32>) -> u32 {
    // counts values using a hashmap
    let mut map = HashMap::new();
    for integer in numbers {
        let count = map.entry(integer).or_insert(0);
        *count += 1;
    }

    // gets the highest count from hashmap values
    let max_count = map.values().cloned().max().unwrap_or(0);

    // get num values by filtering all counts that match max_count
    // this returns multiple values if there are ties
    let max_nums = map.into_iter()
        .filter(|&(_, v)| v == max_count)
        .map(|(&k, _)| k)
        .collect::<Vec<u32>>();

    // return highest number to break the tie
    max_nums.iter().cloned().max().unwrap()
}