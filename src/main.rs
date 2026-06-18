use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

use nd2_rs::{Nd2Index, Nd2Summary, Nd2VersionProbe, discover_nd2_files, fnv1a64};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let command = args
        .next()
        .and_then(|s| s.into_string().ok())
        .unwrap_or_else(|| "help".to_string());
    let paths: Vec<PathBuf> = args.map(PathBuf::from).collect();

    match command.as_str() {
        "inspect" => {
            if paths.is_empty() {
                return Err("usage: nd2-rs inspect FILE.nd2 [FILE.nd2 ...]".into());
            }
            for path in paths {
                inspect_one(&path);
            }
        }
        "scan" => {
            if paths.is_empty() {
                return Err("usage: nd2-rs scan ROOT [ROOT ...]".into());
            }
            scan_roots(&paths)?;
        }
        "versions" => {
            if paths.is_empty() {
                return Err("usage: nd2-rs versions ROOT [ROOT ...]".into());
            }
            scan_versions(&paths)?;
        }
        "bench-read" => {
            if paths.is_empty() {
                return Err("usage: nd2-rs bench-read FILE.nd2 [FILE.nd2 ...]".into());
            }
            for path in paths {
                bench_read_one(&path);
            }
        }
        "dump-pixels" => {
            if paths.len() != 1 {
                return Err("usage: nd2-rs dump-pixels FILE.nd2 > pixels.bin".into());
            }
            dump_pixels(&paths[0])?;
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  nd2-rs inspect FILE.nd2 [FILE.nd2 ...]");
            eprintln!("  nd2-rs scan ROOT [ROOT ...]");
            eprintln!("  nd2-rs versions ROOT [ROOT ...]");
            eprintln!("  nd2-rs bench-read FILE.nd2 [FILE.nd2 ...]");
            eprintln!("  nd2-rs dump-pixels FILE.nd2 > pixels.bin");
        }
    }

    Ok(())
}

fn dump_pixels(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let index = Nd2Index::open(path)?;
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    index.write_pixel_bytes_after_prefix(8, &mut stdout)?;
    Ok(())
}

fn bench_read_one(path: &PathBuf) {
    let t0 = Instant::now();
    match Nd2Index::open(path) {
        Ok(index) => {
            let index_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            match index.read_pixel_bytes_after_prefix(8) {
                Ok(bytes) => {
                    let read_s = t1.elapsed().as_secs_f64();
                    let hash = fnv1a64(&bytes);
                    println!("path\t{}", path.display());
                    println!("status\tok");
                    println!("index_s\t{index_s:.9}");
                    println!("read_s\t{read_s:.9}");
                    println!("bytes\t{}", bytes.len());
                    println!("throughput_MB_s\t{:.6}", bytes.len() as f64 / read_s / 1e6);
                    println!("fnv1a64\t{hash:016x}");
                    println!("plane_count\t{}", index.planes.len());
                }
                Err(err) => {
                    println!("path\t{}", path.display());
                    println!("status\terror");
                    println!("error\t{err}");
                }
            }
        }
        Err(err) => {
            println!("path\t{}", path.display());
            println!("status\terror");
            println!("error\t{err}");
        }
    }
}

fn inspect_one(path: &PathBuf) {
    match Nd2Index::open(path) {
        Ok(index) => {
            println!("path\t{}", index.path.display());
            println!("status\tok");
            println!("variant\t{:?}", index.variant);
            println!(
                "version\t{}",
                index.signature_version.as_deref().unwrap_or("?")
            );
            println!("file_size\t{}", index.file_size);
            println!("filemap_offset\t{}", index.filemap_offset.unwrap_or(0));
            println!("chunk_count\t{}", index.chunk_count);
            println!("plane_count\t{}", index.planes.len());
            if let Some(first) = index.planes.first() {
                println!("first_plane_payload_offset\t{}", first.payload_offset);
                println!("first_plane_payload_len\t{}", first.payload_len);
                println!("first_plane_prefix8\t{:02x?}", first.prefix8);
            }
            if let Some(last) = index.planes.last() {
                println!("last_plane_sequence\t{}", last.sequence);
                println!("last_plane_payload_offset\t{}", last.payload_offset);
                println!("last_plane_payload_len\t{}", last.payload_len);
            }
            println!("chunk_names");
            for (name, count) in &index.chunk_name_counts {
                println!("  {name}\t{count}");
            }
        }
        Err(err) => {
            println!("path\t{}", path.display());
            println!("status\terror");
            println!("error\t{err}");
        }
    }
}

fn scan_versions(paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for root in paths {
        files.extend(discover_nd2_files(root)?);
    }
    files.sort();
    files.dedup();

    let mut ok = 0usize;
    let mut errors = 0usize;
    let mut variants = BTreeMap::<String, usize>::new();
    let mut versions = BTreeMap::<String, usize>::new();
    let mut signature_names = BTreeMap::<String, usize>::new();
    let mut small_lt_1gb = 0usize;
    let mut normal_ge_1gb = 0usize;
    let mut error_examples = Vec::new();

    for path in &files {
        match Nd2VersionProbe::open(path) {
            Ok(probe) => {
                ok += 1;
                if probe.file_size < 1_000_000_000 {
                    small_lt_1gb += 1;
                } else {
                    normal_ge_1gb += 1;
                }
                *variants.entry(format!("{:?}", probe.variant)).or_insert(0) += 1;
                *versions
                    .entry(probe.signature_version.unwrap_or_else(|| "?".to_string()))
                    .or_insert(0) += 1;
                *signature_names.entry(probe.signature_name).or_insert(0) += 1;
            }
            Err(err) => {
                errors += 1;
                if error_examples.len() < 20 {
                    error_examples.push((path.clone(), truncate_error(&err.to_string())));
                }
            }
        }
    }

    println!("files_found\t{}", files.len());
    println!("ok\t{ok}");
    println!("errors\t{errors}");
    println!("small_lt_1gb\t{small_lt_1gb}");
    println!("normal_ge_1gb\t{normal_ge_1gb}");
    print_map("variants", &variants);
    print_map("versions", &versions);
    print_map("signature_names", &signature_names);
    if !error_examples.is_empty() {
        println!("error_examples");
        for (path, err) in error_examples {
            println!("  {}\t{}", path.display(), err);
        }
    }

    Ok(())
}

fn scan_roots(paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for root in paths {
        files.extend(discover_nd2_files(root)?);
    }
    files.sort();
    files.dedup();

    let mut ok = 0usize;
    let mut errors = 0usize;
    let mut variants = BTreeMap::<String, usize>::new();
    let mut versions = BTreeMap::<String, usize>::new();
    let mut plane_counts = BTreeMap::<usize, usize>::new();
    let mut payload_lens = BTreeMap::<u64, usize>::new();
    let mut chunk_bases = BTreeSet::new();
    let mut smallest_ok: Option<(u64, PathBuf)> = None;
    let mut largest_ok: Option<(u64, PathBuf)> = None;
    let mut error_examples = Vec::new();

    for path in &files {
        match Nd2Summary::open(path) {
            Ok(summary) => {
                ok += 1;
                *variants
                    .entry(format!("{:?}", summary.variant))
                    .or_insert(0) += 1;
                *versions
                    .entry(summary.signature_version.unwrap_or_else(|| "?".to_string()))
                    .or_insert(0) += 1;
                *plane_counts.entry(summary.plane_count).or_insert(0) += 1;
                if let Some(payload_len) = summary.first_plane_payload_len {
                    *payload_lens.entry(payload_len).or_insert(0) += 1;
                }
                for name in summary.chunk_name_counts.keys() {
                    chunk_bases.insert(name.clone());
                }
                let size = summary.file_size;
                if smallest_ok
                    .as_ref()
                    .map_or(true, |(smallest, _)| size < *smallest)
                {
                    smallest_ok = Some((size, path.clone()));
                }
                if largest_ok
                    .as_ref()
                    .map_or(true, |(largest, _)| size > *largest)
                {
                    largest_ok = Some((size, path.clone()));
                }
            }
            Err(err) => {
                errors += 1;
                if error_examples.len() < 20 {
                    error_examples.push((path.clone(), truncate_error(&err.to_string())));
                }
            }
        }
    }

    println!("files_found\t{}", files.len());
    println!("ok\t{ok}");
    println!("errors\t{errors}");
    print_map("variants", &variants);
    print_map("versions", &versions);
    print_map("plane_counts", &plane_counts);
    print_map("first_plane_payload_lens", &payload_lens);
    println!("chunk_base_names");
    for name in chunk_bases {
        println!("  {name}");
    }
    if let Some((size, path)) = smallest_ok {
        println!("smallest_ok\t{size}\t{}", path.display());
    }
    if let Some((size, path)) = largest_ok {
        println!("largest_ok\t{size}\t{}", path.display());
    }
    if !error_examples.is_empty() {
        println!("error_examples");
        for (path, err) in error_examples {
            println!("  {}\t{}", path.display(), err);
        }
    }

    Ok(())
}

fn print_map<K: std::fmt::Display>(label: &str, map: &BTreeMap<K, usize>) {
    println!("{label}");
    for (key, count) in map {
        println!("  {key}\t{count}");
    }
}

fn truncate_error(err: &str) -> String {
    let cleaned: String = err
        .chars()
        .map(|ch| {
            if ch.is_ascii_graphic() || ch == ' ' {
                ch
            } else {
                '?'
            }
        })
        .collect();
    const LIMIT: usize = 180;
    if cleaned.len() > LIMIT {
        format!("{}...", &cleaned[..LIMIT])
    } else {
        cleaned
    }
}
