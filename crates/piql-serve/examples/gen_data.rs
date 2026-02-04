//! Generate example parquet files for testing PiQL queries.
//!
//! Run with: cargo run -p piql --example gen_data

use polars::prelude::*;
use std::fs;

fn main() {
    fs::create_dir_all("data").expect("Failed to create data directory");

    let factions = gen_factions();
    let locations = gen_locations();
    let entities = gen_entities();
    let transactions = gen_transactions();

    write_parquet(&factions, "data/factions.parquet");
    write_parquet(&locations, "data/locations.parquet");
    write_parquet(&entities, "data/entities.parquet");
    write_parquet(&transactions, "data/transactions.parquet");

    println!("Generated parquet files in data/");
    println!("  - factions.parquet: {} rows, cols: {:?}", factions.height(), factions.get_column_names());
    println!("  - locations.parquet: {} rows, cols: {:?}", locations.height(), locations.get_column_names());
    println!("  - entities.parquet: {} rows, cols: {:?}", entities.height(), entities.get_column_names());
    println!("  - transactions.parquet: {} rows, cols: {:?}", transactions.height(), transactions.get_column_names());
}

fn write_parquet(df: &DataFrame, path: &str) {
    let mut file = fs::File::create(path).expect("Failed to create file");
    ParquetWriter::new(&mut file)
        .finish(&mut df.clone())
        .expect("Failed to write parquet");
}

fn gen_factions() -> DataFrame {
    let names = [
        "Iron Legion",
        "Shadow Council",
        "Golden Dawn",
        "Crimson Order",
        "Azure Covenant",
        "Emerald Circle",
        "Obsidian Pact",
        "Silver Hand",
        "Void Seekers",
        "Storm Wardens",
    ];
    let alignments = [
        "neutral", "evil", "good", "neutral", "good",
        "good", "evil", "good", "evil", "neutral",
    ];

    df! {
        "faction_id" => (1i64..=10).collect::<Vec<_>>(),
        "faction_name" => names.to_vec(),
        "alignment" => alignments.to_vec(),
    }
    .unwrap()
}

fn gen_locations() -> DataFrame {
    let location_names = [
        "Ironforge", "Darkwood", "Crystalvale", "Thornhaven", "Misthollow",
        "Ashfall", "Frostpeak", "Goldport", "Shadowmere", "Sunridge",
        "Stormwatch", "Deepmine", "Wildheart", "Bonechill", "Embervale",
        "Serenity", "Dreadmoor", "Highcastle", "Grimwater", "Starfall",
        "Dusthaven", "Moonshade", "Ironwall", "Blackmarsh", "Silverbrook",
        "Thunderkeep", "Whisperwind", "Dragoncrest", "Ravenhollow", "Liongate",
        "Serpentine", "Eaglespire", "Wolfden", "Bearhollow", "Foxmead",
        "Owlwatch", "Hawknest", "Crowperch", "Swiftrun", "Stillwater",
        "Brightforge", "Darkspire", "Clearbrook", "Murkwood", "Shimmerdale",
        "Gloomvale", "Radianthill", "Shadowpeak", "Lighthollow", "Duskmeadow",
    ];

    let regions: Vec<&str> = (0..50)
        .map(|i| match i % 4 {
            0 => "north",
            1 => "south",
            2 => "east",
            _ => "west",
        })
        .collect();

    let danger_levels: Vec<i32> = (0..50).map(|i| (i % 10) + 1).collect();

    df! {
        "location_id" => (1i64..=50).collect::<Vec<_>>(),
        "location_name" => location_names.to_vec(),
        "region" => regions,
        "danger_level" => danger_levels,
    }
    .unwrap()
}

fn gen_entities() -> DataFrame {
    let n = 300;

    let entity_ids: Vec<i64> = (1..=n as i64).collect();

    let names: Vec<String> = (1..=n).map(|i| format!("Entity_{:03}", i)).collect();

    let types: Vec<&str> = (0..n)
        .map(|i| match i % 4 {
            0 => "merchant",
            1 => "producer",
            2 => "warrior",
            _ => "mage",
        })
        .collect();

    let faction_ids: Vec<i64> = (0..n).map(|i| (i % 10) as i64 + 1).collect();

    let location_ids: Vec<i64> = (0..n).map(|i| (i % 50) as i64 + 1).collect();

    let levels: Vec<i32> = (0..n).map(|i| (i % 50) as i32 + 1).collect();

    let created_ticks: Vec<i64> = (0..n).map(|i| (i / 10) as i64 + 1).collect();

    df! {
        "entity_id" => entity_ids,
        "name" => names,
        "type" => types,
        "faction_id" => faction_ids,
        "location_id" => location_ids,
        "level" => levels,
        "created_tick" => created_ticks,
    }
    .unwrap()
}

fn gen_transactions() -> DataFrame {
    let n = 500;

    let tx_ids: Vec<i64> = (1..=n as i64).collect();

    // Spread transactions across 50 ticks
    let ticks: Vec<i64> = (0..n).map(|i| (i / 10) as i64 + 1).collect();

    // Cycle through entities for from/to
    let from_ids: Vec<i64> = (0..n).map(|i| (i % 300) as i64 + 1).collect();
    let to_ids: Vec<i64> = (0..n).map(|i| ((i + 50) % 300) as i64 + 1).collect();

    // Amounts from 10 to 1000
    let amounts: Vec<i64> = (0..n).map(|i| ((i % 100) + 1) as i64 * 10).collect();

    let tx_types: Vec<&str> = (0..n)
        .map(|i| match i % 4 {
            0 => "trade",
            1 => "tax",
            2 => "reward",
            _ => "loot",
        })
        .collect();

    df! {
        "tx_id" => tx_ids,
        "tick" => ticks,
        "from_entity_id" => from_ids,
        "to_entity_id" => to_ids,
        "amount" => amounts,
        "tx_type" => tx_types,
    }
    .unwrap()
}
