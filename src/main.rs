use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use dialoguer::{Input, Select, theme::ColorfulTheme};
use hound;
use rustfft::Fft;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    let mut database: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
    let mut song_ids: HashMap<String, usize> = HashMap::new();
    let mut song_names: Vec<String> = Vec::new();
    loop {
        let options = &["add", "match"];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Choose your Action")
            .default(0)
            .items(&options[..])
            .interact()
            .unwrap();

        let action = options[selection];

        if action == "add" {
            let path: String = Input::new()
                .with_prompt("Enter Path to save to ")
                .interact_text()
                .unwrap();
            let song_id = if let Some(&id) = song_ids.get(&path) {
                id
            } else {
                let id = song_names.len();
                song_names.push(path.clone());
                song_ids.insert(path.clone(), id);
                id
            };
            upload_audio_and_save(path.clone());
            thread::sleep(Duration::from_secs(15));
            let magnitudes_per_frame = fourier_transform_spectrogramm(path.clone());
            let constellationmap = create_constellation_map(magnitudes_per_frame);
            let fingerprints = generate_finger_prints(constellationmap);
            add_song_to_database(&mut database, song_id, fingerprints);
        } else {
            let magnitudes_per_frame = fourier_transform_spectrogramm("input.wav".to_string());
            let constellationmap = create_constellation_map(magnitudes_per_frame);
            let fingerprints = generate_finger_prints(constellationmap);

            let mut song_matches: HashMap<usize, Vec<i32>> = HashMap::new();

            for (hash, time) in fingerprints {
                if let Some(matches) = database.get(&hash) {
                    for (song_id, time_in_song) in matches {
                        let time_diff = *time_in_song as i32 - time as i32;

                        song_matches
                            .entry(*song_id)
                            .or_insert(Vec::new())
                            .push(time_diff);
                    }
                }
            }
            let mut best_song: Option<String> = None;
            let mut best_count = 0;

            for (song_id, time_diffs) in song_matches {
                let song_name = &song_names[song_id];
                // Zähle wie  die gleiche Differenz vorkommt
                let mut diff_counts: HashMap<i32, usize> = HashMap::new();

                for diff in time_diffs {
                    *diff_counts.entry(diff).or_insert(0) += 1;
                }

                if let Some(max_count) = diff_counts.values().max() {
                    if *max_count > best_count {
                        best_count = *max_count;
                        best_song = Some(song_name.to_string());
                    }
                }
            }

            if best_count >= 5 {
                println!(
                    "Match found: {:?} with {} aligned matches",
                    best_song, best_count
                );
            } else {
                println!("No match found (best: {} matches)", best_count);
            }
        }
    }
}

fn add_song_to_database(
    database: &mut HashMap<u64, Vec<(usize, usize)>>,
    song_id: usize,
    fingerprints: Vec<(u64, usize)>,
) {
    for (hash, time) in fingerprints {
        database
            .entry(hash)
            .or_insert(Vec::new())
            .push((song_id, time));
    }
}

fn upload_audio_and_save(path: String) {
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .expect("no Input device available");

    // save all the samples
    let samples = Arc::new(Mutex::new(Vec::new()));
    let samples_clone: Arc<Mutex<Vec<f32>>> = samples.clone();
    let config: cpal::SupportedStreamConfig = device.default_input_config().unwrap();
    let stream = device
        .build_input_stream(
            &config.into(),
            move |data: &[f32], _| {
                let mut samples_lock = samples_clone.lock().unwrap();
                samples_lock.extend_from_slice(data); // copy all samples
            },
            move |err| println!("{}", err),
            None,
        )
        .unwrap();
    // start the stream
    stream.play().unwrap();

    // stop after 10 sec
    thread::sleep(Duration::from_secs(10));

    // stop the stream
    drop(stream);

    let samples_vec = samples.lock().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    for &sample in samples_vec.iter() {
        let amplitude = (sample * i16::MAX as f32) as i16;
        writer.write_sample(amplitude).unwrap();
    }

    writer.finalize().unwrap();
}

fn fourier_transform_spectrogramm(path: String) -> Vec<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).unwrap();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s: Result<i16, hound::Error>| s.unwrap() as f32)
        .collect();

    let mut planner = FftPlanner::new();

    let frame_size = 1024; // typical frame size
    let hop_size = 512;

    let mut frames: Vec<&[f32]> = Vec::new();

    let mut i = 0;

    while i + frame_size <= samples.len() {
        frames.push(&samples[i..i + frame_size]);
        i += hop_size;
    }

    let mut magnitudes_per_frame: Vec<Vec<f32>> = Vec::new();

    let fft: Arc<dyn Fft<_>> = planner.plan_fft_forward(frame_size);

    for frame in frames {
        let mut buffer: Vec<Complex<f32>> =
            frame.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();

        fft.process(&mut buffer);

        let mags: Vec<f32> = buffer
            .iter()
            .map(|c| (c.re.powi(2) + c.im.powi(2)).sqrt())
            .collect();
        magnitudes_per_frame.push(mags)
    }

    magnitudes_per_frame
}

fn create_constellation_map(magnitudes_per_frame: Vec<Vec<f32>>) -> Vec<(usize, usize, f32)> {
    let mut peaks: Vec<(usize, usize, f32)> = Vec::new();
    let threshold = 0.1;

    for (time_idx, frame) in magnitudes_per_frame.iter().enumerate() {
        let mut peaks_in_frame: Vec<(usize, f32)> = frame
            .iter()
            .enumerate()
            .filter(|(_, mag)| **mag > threshold)
            .map(|(i, &mag)| (i, mag))
            .collect();

        peaks_in_frame.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks_in_frame.truncate(5);

        for (freq_index, magnitude) in peaks_in_frame {
            if magnitude > threshold && is_peak(&magnitudes_per_frame, time_idx, freq_index) {
                peaks.push((time_idx, freq_index, magnitude))
            };
        }
    }

    peaks
}

fn is_peak(magnitudes_per_frame: &Vec<Vec<f32>>, time_idx: usize, freq_idx: usize) -> bool {
    let current = magnitudes_per_frame[time_idx][freq_idx];

    for t in time_idx.saturating_sub(1)..=time_idx.saturating_add(1) {
        if t >= magnitudes_per_frame.len() {
            continue;
        }

        for f in freq_idx.saturating_sub(1)..=freq_idx.saturating_add(1) {
            if f >= magnitudes_per_frame[t].len() {
                continue;
            }

            if t == time_idx && f == freq_idx {
                continue;
            }

            if magnitudes_per_frame[t][f] > current {
                return false;
            }
        }
    }

    true
}

fn generate_finger_prints(peaks: Vec<(usize, usize, f32)>) -> Vec<(u64, usize)> {
    let mut finger_prints = Vec::new();

    for i in 0..peaks.len() {
        let (time_anchor, freq_anchor, _) = peaks[i];

        for j in (i + 1)..peaks.len() {
            let (time_target, freq_target, _) = peaks[j];
            let delta_time = time_target - time_anchor;

            if delta_time >= 5 && delta_time <= 40 {
                let hash = createhash(freq_anchor, freq_target, delta_time);
                finger_prints.push((hash, time_anchor));
            }
            if delta_time > 40 {
                break;
            }
        }
    }
    finger_prints
}

fn createhash(freq1: usize, freq2: usize, delta_time: usize) -> u64 {
    ((freq1 as u64) << 32) | ((freq2 as u64) << 16) | (delta_time as u64)
}

/*
    How Shazam Works

    1 Channel Mono
    Sample Rate 44100 Hz

*/
