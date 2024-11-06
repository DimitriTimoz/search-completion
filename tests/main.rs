use std::fs::File;

use anyhow::{Error as E, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use search_completion::*;
use candle_transformers::models::{bert::{self, DTYPE}, mimi::candle_nn::VarBuilder};
use candle_core as candle;
use tokenizers::Tokenizer;
#[test]
fn test() {
    let default_model = "huawei-noah/TinyBERT_General_4L_312D".to_string();
    let default_revision = "main".to_string();

    let repo = Repo::with_revision(default_model, RepoType::Model, default_revision);

    let api = Api::new().unwrap();
    let api = api.repo(repo.clone());
    let (config_filename, weights_filename) = {
        let api = Api::new().unwrap();
        let api = api.repo(repo);
        let config = api.get("config.json").unwrap();
        let weights = api.get("pytorch_model.bin").unwrap();
        (config, weights)
    };
    let config = std::fs::read_to_string(&config_filename).unwrap();

    let path = "/Users/dimitri/.cache/huggingface/hub/models--huawei-noah--TinyBERT_General_4L_312D/snapshots/4d2f8cf0079689fb3a1e27972aa29ec4c171ceed/model.safetensors"
        .to_string();
    let file = File::open(path.clone()).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();


    let config: bert::Config = serde_json::from_str(&config).unwrap();
    let device = candle::Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DTYPE, &device).unwrap() };
    let model = bert::BertModel::load(vb, &config).unwrap();

    println!("tensors: {:?}", tensors.names());
    //hub_load_safetensors(&api, config_filename.as_os_str().to_str().unwrap()).unwrap();
}
