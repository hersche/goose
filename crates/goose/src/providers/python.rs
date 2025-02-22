use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::utils::{get_model};
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use anyhow::Result;
use async_trait::async_trait;
use mcp_core::tool::Tool;
use serde_json::Value;
use std::env;
use std::path::Path;
use std::process::Stdio;
use tokio::process::Command;

pub const PYTHON_PROVIDER_CMD: &str = "./venv/bin/python ./wrapper.py";
pub const PYTHON_PROVIDER_DEFAULT_MODEL: &str = "gemini-2.0-flash-thinking-exp";
pub const PYTHON_PROVIDER_KNOWN_MODELS: &[&str] = &[PYTHON_PROVIDER_DEFAULT_MODEL];
pub const PYTHON_PROVIDER_DOC_URL: &str = "https://ai.google.dev/gemini-api/docs";

#[derive(serde::Serialize)]
pub struct PythonProvider {
    #[serde(skip)]
    script_cmd: String,
    model: ModelConfig,
}

impl Default for PythonProvider {
    fn default() -> Self {
        let model = ModelConfig::new(PYTHON_PROVIDER_DEFAULT_MODEL.to_string());
        PythonProvider::from_env(model).expect("Failed to initialize Python provider")
    }
}

impl PythonProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let script_cmd = env::var("PYTHON_PROVIDER_CMD")
            .unwrap_or_else(|_| PYTHON_PROVIDER_CMD.to_string());
        Ok(Self { script_cmd, model })
    }

    async fn ensure_venv(&self) -> Result<(), ProviderError> {
        let venv_dir = "venv";
        if !Path::new(venv_dir).exists() {
            let create_status = Command::new("/usr/bin/python3")
                .args(&["-m", "venv", venv_dir])
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .output()
                .await
                .map_err(|e| ProviderError::RequestFailed(format!("Failed to create venv: {}", e)))?;
            if !create_status.status.success() {
                let stderr = String::from_utf8_lossy(&create_status.stderr);
                return Err(ProviderError::RequestFailed(format!("Venv creation failed: {}", stderr)));
            }
            let pip_path = format!("{}/bin/pip", venv_dir);
            let dependencies = ["google-genai"];
            let mut pip_args = vec!["install".to_string()];
            for dep in dependencies.iter() {
                pip_args.push(dep.to_string());
            }
            let install_status = Command::new(&pip_path)
                .args(&pip_args)
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .output()
                .await
                .map_err(|e| ProviderError::RequestFailed(format!("Failed to install dependencies: {}", e)))?;
            if !install_status.status.success() {
                let stderr = String::from_utf8_lossy(&install_status.stderr);
                return Err(ProviderError::RequestFailed(format!("Dependency installation failed: {}", stderr)));
            }
        }
        println!("Check for venv ended");
        Ok(())
    }

    async fn execute(&self, prompt: &str) -> Result<Value, ProviderError> {
        self.ensure_venv().await?;
        // Split the command into the python interpreter and script path.
        let parts: Vec<&str> = self.script_cmd.split_whitespace().collect();
        let python_exe = parts.get(0).unwrap_or(&"./venv/bin/python");
        // Use the script path from the constant as the first argument.
        let script_path = parts.get(1).unwrap_or(&"./wrapper.py");
        let mut args: Vec<String> = Vec::new();
        args.push(script_path.to_string());
        args.push("--prompt".to_string());
        args.push(prompt.to_string());
        if let Ok(api_key) = env::var("GOOGLE_API_KEY") {
            args.push("--api-key".to_string());
            args.push(api_key);
        }
        println!("execute {} with args {:?}", python_exe, args);
        let output = Command::new(python_exe)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to execute command: {}", e)))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ProviderError::RequestFailed(format!("Command failed: {}", stderr)));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        serde_json::from_str(&stdout)
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to parse JSON: {}", e)))
    }
}

#[async_trait]
impl Provider for PythonProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "python",
            "Python Provider",
            "Provider using a command-line Python script (with venv support) for the Gemini API",
            PYTHON_PROVIDER_DEFAULT_MODEL,
            PYTHON_PROVIDER_KNOWN_MODELS.iter().map(|&s| s.to_string()).collect(),
            PYTHON_PROVIDER_DOC_URL,
            vec![ConfigKey::new("PYTHON_PROVIDER_CMD", false, false, Some(PYTHON_PROVIDER_CMD))],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(skip(self, system, messages, tools))]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(
            &self.model,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        )?;
        let prompt = serde_json::to_string(&payload)
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let response = self.execute(&prompt).await?;
        let message = response_to_message(response.clone())?;
        println!("m3ssage {}", response);
        let usage = match get_usage(&response) {
            Ok(u) => u,
            Err(ProviderError::UsageError(e)) => {
                tracing::debug!("Usage error: {}", e);
                Usage::default()
            }
            Err(e) => return Err(e),
        };
        let model = get_model(&response);
        Ok((message, ProviderUsage::new(model, usage)))
    }
}
