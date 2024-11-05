use anyhow::{Error as E, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use search_completion::*;
use tokenizers::{PaddingParams, Tokenizer};

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
    println!("config: {:?}", config);
    hub_load_safetensors(&api, config_filename.as_os_str().to_str().unwrap()).unwrap();
}
