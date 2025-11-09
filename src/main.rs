use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    upload_audio_and_save();
}

fn upload_audio_and_save() {
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

    let mut writer = hound::WavWriter::create("input.wav", spec).unwrap();

    for &sample in samples_vec.iter() {
        let amplitude = (sample * i16::MAX as f32) as i16;
        writer.write_sample(amplitude).unwrap();
    }

    writer.finalize().unwrap();
}

fn fourier_transform_spectrogramm() {
    let mut reader = hound::WavReader::open("input.wav").unwrap();
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32).collect();

    let mut planner = FftPlanner::new();

    let fft = planner.plan_fft_forward(samples.len());

    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();

    fft.process(&mut buffer);
}

/*
    How Shazam Works

    1 Channel Mono
    Sample Rate 44100 Hz

*/
