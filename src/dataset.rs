use std::error::Error;

#[derive(Debug)]
pub struct Dataset {
    pub features: Vec<Vec<String>>,
    pub targets: Vec<String>,
    pub num_samples: usize,
    pub num_features: usize,
}

impl Dataset {
    pub fn load_csv_winrate(filepath: &str) -> Result<Self, Box<dyn Error>> {
        println!("Loading LoL dataset from: {}", filepath);

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(filepath)?; // ? handles the error if the file doesn't exist.

        let headers = reader.headers()?.clone();

        let winrate_idx = headers
            .iter()
            .position(|h| h == "Win_rate")
            .ok_or_else(|| "Win_rate column not found in CSV headers")?;
        
        let excluded_cols = [ //useless for our model so we exclude it
            "Player",
            "Country",
            "SeasonYear",
            "SourceURL",
            "Split",      
            "Team",       
            "Time",
        ];

        let mut excluded_indices = Vec::new(); //index #s for useless column
        for (i, h) in headers.iter().enumerate() {
            if excluded_cols.contains(&h.as_ref()) || i == winrate_idx {
                excluded_indices.push(i);
            }
        }

        let mut features: Vec<Vec<String>> = Vec::new();
        let mut targets: Vec<String> = Vec::new();
        let mut count = 0;
        //loop through every row
        for record_result in reader.records() {
            let record = record_result?;
            let mut row_features = Vec::new();

            let str = &record[winrate_idx];
            let clean = str.trim().trim_end_matches('%'); //remove so parse to float

            let val: f64 = clean.parse().unwrap();

            let label = if val > 50.0 { "High" } else { "Low" };
            targets.push(label.to_string());
            // If this column index is in the excluded list, skip
            for (idx, field) in record.iter().enumerate() {
                if excluded_indices.contains(&idx) {
                    continue;
                }
                row_features.push(field.to_string());
            }
            // Update the feature count based on the last row
            count = row_features.len();
            features.push(row_features);
        }
        
        let sample = features.len();

        Ok(Dataset {
            features: features,
            targets: targets,
            num_samples: sample,
            num_features: count,
        })
    }
}


pub fn get_player_name(filepath: &str, row_index: usize) -> Option<String> {
    let mut reader = match csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(filepath) {
            Ok(r) => r,
            Err(_) => return None,
        };

    for (i, record) in reader.records().enumerate() {
        if i == row_index {
            let record = record.unwrap(); 
            return Some(record.get(0).unwrap().to_string());
        }
    }
    
    //If loop finished never found the index
    None
}