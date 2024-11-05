use candle_core as candle;
use hf_hub;

use serde::Serialize;
use serde_derive::Deserialize;
use serde_json::Value;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    #[serde(rename = "attention_probs_dropout_prob")]
    pub attention_probs_dropout_prob: f64,
    pub cell: Cell,
    #[serde(rename = "model_type")]
    pub model_type: String,
    #[serde(rename = "emb_size")]
    pub emb_size: i64,
    #[serde(rename = "hidden_act")]
    pub hidden_act: String,
    #[serde(rename = "hidden_dropout_prob")]
    pub hidden_dropout_prob: f64,
    #[serde(rename = "hidden_size")]
    pub hidden_size: i64,
    #[serde(rename = "initializer_range")]
    pub initializer_range: f64,
    #[serde(rename = "intermediate_size")]
    pub intermediate_size: i64,
    #[serde(rename = "max_position_embeddings")]
    pub max_position_embeddings: i64,
    #[serde(rename = "num_attention_heads")]
    pub num_attention_heads: i64,
    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: i64,
    #[serde(rename = "pre_trained")]
    pub pre_trained: String,
    pub structure: Vec<Value>,
    #[serde(rename = "type_vocab_size")]
    pub type_vocab_size: i64,
    #[serde(rename = "vocab_size")]
    pub vocab_size: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cell {
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> candle::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let mut safetensors_files = std::collections::HashSet::new();
    safetensors_files.insert("pytorch_model.bin");
              
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<candle::Result<Vec<_>>>()?;
    println!("safetensors_files: {:?}", safetensors_files);
    let device = candle::Device::Cpu;
    let a = candle::safetensors::load(safetensors_files.first().unwrap(), &device)?;
    Ok(safetensors_files)
}
