use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;

mod dataset;
mod tree;

use dataset::{Dataset, get_player_name}; 
use tree::{build_tree, Node};


fn main() -> Result<(), Box<dyn Error>> {
    println!("Decision Tree in Rust for LOL prediction");
    
    let dataset = Dataset::load_csv_winrate("../lck_player_stats_2021_2025.csv")?; 

    // shuffle indices for 70/30 train/test split
    let mut indices: Vec<usize> = (0..dataset.num_samples).collect();
    indices.shuffle(&mut thread_rng());

    let split_at = (0.7 * dataset.num_samples as f64) as usize;
    let (train_idx, test_idx) = indices.split_at(split_at);

    let max_depth = 5;

    let tree_root = build_tree(&dataset, train_idx, 0, max_depth);


    let mut correct = 0;
    for &i in test_idx { // Check accuracy on the test data
        let x = &dataset.features[i];
        let true_label = &dataset.targets[i];
        let pred = tree_root.predict(x);
        
        if &pred == true_label {
            correct += 1;
        }
    }

    let acc = correct as f64 / test_idx.len() as f64;
    println!("--- Results ---");
    println!("Accuracy: {:.3}", acc); 
    // Print a few test predictions to check sanity
    println!("predictions from test set:");
    for &i in test_idx.iter().take(5) {
        let x = &dataset.features[i];
        let true_label = &dataset.targets[i];
        let pred = tree_root.predict(x);
        println!("{i}: actual = {true_label}, predicted = {pred}");
    }

    // Manual check for specific rows
    println!("Checking player 56, 57, 58");
    let filepath = "lck_player_stats_2021_2025.csv"; 
    for &i in &[56usize, 57usize, 58usize] {
        if i >= dataset.num_samples {
            println!("It doesn't exist");
            continue;
        }

        let x = &dataset.features[i];
        let true_label = &dataset.targets[i];
        let pred = tree_root.predict(x);

        let player = get_player_name(filepath, i).unwrap_or("UNKNOWN PLAYER".to_string());
        println!("Row {i} ({player}): REAL = {true_label}, PREDICTED = {pred}");
    }
    // Final check on class balance
    let count_high = dataset.targets.iter().filter(|x| x == &&"High").count();
    let count_low = dataset.targets.iter().filter(|x| x == &&"Low").count();
    println!("Total Counts: High: {}, Low: {}", count_high, count_low);

    Ok(())
}