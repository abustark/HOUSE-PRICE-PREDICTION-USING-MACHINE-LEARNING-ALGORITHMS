use clap::{Parser, Subcommand};
use hpp_core::{clean_area_type, extract_bhk, parse_availability, parse_sqft};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "hpp-cli")]
#[command(about = "High-performance parallel real-estate ETL & cleaning engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest and print statistical analysis of the raw real-estate dataset
    Stats {
        /// Path to raw Bengaluru CSV dataset
        #[arg(short, long, default_value = "data/Bengaluru_House_Data.csv")]
        input: PathBuf,
    },
    /// Clean and export normalized listings to CSV
    Clean {
        /// Path to raw Bengaluru CSV dataset
        #[arg(short, long, default_value = "data/Bengaluru_House_Data.csv")]
        input: PathBuf,
        /// Path to save cleaned dataset
        #[arg(short, long, default_value = "data/cleaned_bengaluru.csv")]
        output: PathBuf,
    },
    /// Benchmark raw parsing and unit-conversion throughput across CPU cores
    Benchmark {
        /// Path to raw dataset
        #[arg(short, long, default_value = "data/Bengaluru_House_Data.csv")]
        input: PathBuf,
        /// Number of benchmark passes
        #[arg(short, long, default_value_t = 10)]
        iterations: u32,
    },
}

#[derive(Debug, Deserialize)]
struct RawRecord {
    area_type: Option<String>,
    availability: Option<String>,
    location: Option<String>,
    size: Option<String>,
    society: Option<String>,
    total_sqft: Option<String>,
    bath: Option<String>,
    balcony: Option<String>,
    price: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CleanRecord {
    area_type: String,
    location: String,
    society: String,
    total_sqft: f64,
    bhk: f64,
    bath: f64,
    balcony: f64,
    completion_year: u32,
    price_lakh: f64,
    price_log: f64,
}

fn read_raw_records(path: &PathBuf) -> Result<Vec<RawRecord>, csv::Error> {
    let file = File::open(path).map_err(|e| csv::Error::from(std::io::Error::new(std::io::ErrorKind::NotFound, e)))?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        records.push(result?);
    }
    Ok(records)
}

fn process_raw_record(r: &RawRecord) -> Option<CleanRecord> {
    let price: f64 = r.price.as_ref()?.trim().parse().ok()?;
    if price <= 0.0 {
        return None;
    }

    let sqft = parse_sqft(r.total_sqft.as_ref()?)?;
    if sqft < 100.0 || sqft > 10000.0 {
        return None;
    }

    let bhk = extract_bhk(r.size.as_ref()?)?;
    let area_type = clean_area_type(r.area_type.as_ref()?)?;

    let bath: f64 = r.bath.as_ref().and_then(|s| s.trim().parse().ok()).unwrap_or(2.0);
    let balcony: f64 = r.balcony.as_ref().and_then(|s| s.trim().parse().ok()).unwrap_or(1.0);
    let completion_year = r.availability.as_ref().map(|s| parse_availability(s)).unwrap_or(0);

    let location = r.location.as_ref().map(|s| s.trim().to_string()).unwrap_or_else(|| "Unknown".to_string());
    let society = r.society.as_ref().map(|s| s.trim().to_string()).unwrap_or_default();

    Some(CleanRecord {
        area_type,
        location,
        society,
        total_sqft: sqft,
        bhk,
        bath,
        balcony,
        completion_year,
        price_lakh: price,
        price_log: (price + 1.0).ln(),
    })
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Stats { input } => {
            println!("╔════════════════════════════════════════════════════════════╗");
            println!("║     Bengaluru Real Estate High-Speed Rust ETL Engine       ║");
            println!("╚════════════════════════════════════════════════════════════╝");
            println!("Reading: {}", input.display());

            let t0 = Instant::now();
            let raw_records = match read_raw_records(&input) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Failed to read CSV: {}", e);
                    std::process::exit(1);
                }
            };
            let io_duration = t0.elapsed();

            let t_process = Instant::now();
            let clean_records: Vec<CleanRecord> = raw_records
                .par_iter()
                .filter_map(process_raw_record)
                .collect();
            let process_duration = t_process.elapsed();

            let total_raw = raw_records.len();
            let total_clean = clean_records.len();
            let dropped = total_raw - total_clean;

            let mut prices: Vec<f64> = clean_records.iter().map(|r| r.price_lakh).collect();
            prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median_price = if prices.is_empty() {
                0.0
            } else {
                prices[prices.len() / 2]
            };

            let avg_sqft = if clean_records.is_empty() {
                0.0
            } else {
                clean_records.iter().map(|r| r.total_sqft).sum::<f64>() / clean_records.len() as f64
            };

            let mut area_counts: HashMap<String, usize> = HashMap::new();
            for r in &clean_records {
                *area_counts.entry(r.area_type.clone()).or_insert(0) += 1;
            }

            println!("\nDataset Summary:");
            println!("  Total Raw Rows:        {}", total_raw);
            println!("  Clean Valid Listings:  {} ({:.1}%)", total_clean, (total_clean as f64 / total_raw as f64) * 100.0);
            println!("  Dropped/Filtered Rows: {} ({:.1}%)", dropped, (dropped as f64 / total_raw as f64) * 100.0);
            println!("  Median Price:          ₹{:.2} Lakh", median_price);
            println!("  Average Total Sqft:    {:.1} sqft", avg_sqft);
            println!("\nArea Type Breakdown:");
            for (area, count) in area_counts {
                println!("  • {:20} -> {} ({:.1}%)", area, count, (count as f64 / total_clean as f64) * 100.0);
            }

            println!("\nRust Performance Metrics:");
            println!("  CSV Read Time:         {:.2?}", io_duration);
            println!("  Parallel Cleaning Time: {:.2?}", process_duration);
            println!("  Throughput:            {:.0} rows/sec", total_raw as f64 / process_duration.as_secs_f64());
        }

        Commands::Clean { input, output } => {
            println!("Cleaning {} -> {}", input.display(), output.display());
            let t0 = Instant::now();
            let raw_records = read_raw_records(&input).expect("Failed to read raw CSV");
            let clean_records: Vec<CleanRecord> = raw_records
                .par_iter()
                .filter_map(process_raw_record)
                .collect();

            let mut wtr = csv::Writer::from_path(&output).expect("Failed to create output CSV");
            for rec in &clean_records {
                wtr.serialize(rec).expect("Failed to write record");
            }
            wtr.flush().expect("Failed to flush CSV");

            println!("Exported {} clean rows in {:.2?}", clean_records.len(), t0.elapsed());
        }

        Commands::Benchmark { input, iterations } => {
            println!("Running Benchmark on {} ({} iterations)...", input.display(), iterations);
            let raw_records = read_raw_records(&input).expect("Failed to read raw CSV");
            println!("Loaded {} rows. Commencing multi-core parallel parsing passes...\n", raw_records.len());

            let mut durations = Vec::new();
            for i in 1..=iterations {
                let start = Instant::now();
                let clean_count = raw_records
                    .par_iter()
                    .filter_map(process_raw_record)
                    .count();
                let elapsed = start.elapsed();
                let rows_per_sec = raw_records.len() as f64 / elapsed.as_secs_f64();
                println!("  Iteration {:2}: {:.2?} | Clean rows: {} | Throughput: {:.0} rows/sec", i, elapsed, clean_count, rows_per_sec);
                durations.push(elapsed);
            }

            let avg_duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
            let avg_throughput = raw_records.len() as f64 / avg_duration.as_secs_f64();
            println!("\nBenchmark Results:");
            println!("  Average Parallel Clean Time: {:.2?}", avg_duration);
            println!("  Average Parsing Throughput:  {:.0} rows/second", avg_throughput);
        }
    }
}
