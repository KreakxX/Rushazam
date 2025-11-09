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
    // out Database for saving all the hashes and the id and times multiple songs can have the same hash
    let mut database: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();

    // saving the song ids for better perfirmacne instead of saving strings later on convert back to give out the real
    let mut song_ids: HashMap<String, usize> = HashMap::new();
    // song names
    let mut song_names: Vec<String> = Vec::new();
    loop {
        // basic CLI functions
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

            // check if its already in there than use the id or else create the id aslong as the name of the string is
            let song_id = if let Some(&id) = song_ids.get(&path) {
                id
            } else {
                let id = song_names.len();
                song_names.push(path.clone());
                song_ids.insert(path.clone(), id);
                id
            };

            // now record the audio and save it as a wav file
            upload_audio_and_save(path.clone());
            thread::sleep(Duration::from_secs(15));

            // afterwards calculate the fourier_transform spectogramm neccessary for getting all the frequency spikes by calculating and generating a graph with frequency and time
            let magnitudes_per_frame = fourier_transform_spectrogramm(path.clone());

            // generate the constellationMap a.k.a the peaks only use the top N because of perfroamcen and memory issues
            let constellationmap = create_constellation_map(magnitudes_per_frame);

            // lastly generate all the fingerprints for each song with hashes
            let fingerprints = generate_finger_prints(constellationmap);

            // and adding the song to the databaes
            add_song_to_database(&mut database, song_id, fingerprints);
        } else {
            // same thing here just calculating the fingerprints of the recording
            let magnitudes_per_frame = fourier_transform_spectrogramm("input.wav".to_string());
            let constellationmap = create_constellation_map(magnitudes_per_frame);
            let fingerprints = generate_finger_prints(constellationmap);

            let mut song_matches: HashMap<usize, Vec<i32>> = HashMap::new();

            // and now match with the one from the database
            for (hash, time) in fingerprints {
                // go along the results and check the DB for matches
                if let Some(matches) = database.get(&hash) {
                    // go through every single match
                    for (song_id, time_in_song) in matches {
                        // and calculate the time_diff neccessary for checking identity for example if you try to match only a part of a song with the full song you dont have the same times but if you have the same time diffs like a Song has mutliple hashes for the peaks and if for the hashes that are similiar we take both the times for song 1 for example 100 and for song 2 for exmaple 200 and we see the diff is 100 and we check ok for hash 2 the diff is also 100 we see that the songs are matching
                        let time_diff = *time_in_song as i32 - time as i32;

                        // now lets add them
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
                // Zähle wie oft die gleiche Differenz vorkommt
                let mut diff_counts: HashMap<i32, usize> = HashMap::new();

                // foreach diff in time Diffs check the amount of the time diff
                for diff in time_diffs {
                    *diff_counts.entry(diff).or_insert(0) += 1;
                }

                // and where its the most we update the max_count because its better now
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

// nothing special just adding the key and value to the DB
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

// function for recording and creating a Wav File from the Input device
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

// complicated Stuff from shazam, saying it simple its just calculating the intensity of the frequenzy at a give time marking it brighter like a spectrogramm, In Song for example words are bright because the intensity of the frequency is higher at the give time but when its like a pause for example its dark and not bright
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

// now calculating the peaks from the magnitudes per fraem
fn create_constellation_map(magnitudes_per_frame: Vec<Vec<f32>>) -> Vec<(usize, usize, f32)> {
    let mut peaks: Vec<(usize, usize, f32)> = Vec::new();
    let threshold = 0.1;

    for (time_idx, frame) in magnitudes_per_frame.iter().enumerate() {
        let mut peaks_in_frame: Vec<(usize, f32)> = frame
            .iter()
            .enumerate() // also keep only the ones where the mag is smaller than the threshold like all the unneccessary stuff we onyl want to keep the important
            .filter(|(_, mag)| **mag > threshold)
            .map(|(i, &mag)| (i, mag))
            .collect();

        peaks_in_frame.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks_in_frame.truncate(10);

        // just map through them and take the 10 highest ones so we save memory/RAM

        // therefore loop throught the top 10 and calulate if its a Peak by using the helper function and if so push it to peaks
        for (freq_index, magnitude) in peaks_in_frame {
            if magnitude > threshold && is_peak(&magnitudes_per_frame, time_idx, freq_index) {
                peaks.push((time_idx, freq_index, magnitude))
            };
        }
    }

    peaks
}

// helper function just go through them by checking the  one before and the one afterwards if its bigger by comparing it else its the highgest retur true
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

            if delta_time >= 5 && delta_time <= 50 {
                let hash = createhash(freq_anchor, freq_target, delta_time);
                finger_prints.push((hash, time_anchor));
            }
            if delta_time > 50 {
                break;
            }
        }
    }
    finger_prints
}

// simple function adding all values together to create a hash
fn createhash(freq1: usize, freq2: usize, delta_time: usize) -> u64 {
    ((freq1 as u64) << 32) | ((freq2 as u64) << 16) | (delta_time as u64)
}

/*
    How Shazam Works

    1 Channel Mono
    Sample Rate 44100 Hz

*/
