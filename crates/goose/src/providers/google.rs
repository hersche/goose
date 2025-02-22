use super::errors::ProviderError;
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage};
use crate::providers::formats::google::{create_request, get_usage, response_to_message};
use crate::providers::utils::{
    emit_debug_trace, handle_response_google_compat, unescape_json_values,
};
use anyhow::Result;
use async_trait::async_trait;
use mcp_core::tool::Tool;
use reqwest::{Client, StatusCode};
use serde_json::Value;
use std::time::Duration;
use url::Url;
use rand::Rng;

pub const GOOGLE_API_HOST: &str = "https://generativelanguage.googleapis.com";
pub const GOOGLE_DEFAULT_MODEL: &str = "gemini-2.0-flash";
pub const GOOGLE_KNOWN_MODELS: &[&str] = &[
    "models/gemini-1.5-pro-latest",
    "models/gemini-1.5-pro",
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite-preview-02-05",
    "models/gemini-2.0-flash-thinking-exp-01-21",
    "models/gemini-2.0-pro-exp-02-05",
];

pub const GOOGLE_DOC_URL: &str = "https://ai.google/get-started/our-models/";

#[derive(Debug, serde::Serialize)]
pub struct GoogleProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    api_key: String,
    model: ModelConfig,
}

impl Default for GoogleProvider {
    fn default() -> Self {
        let model = ModelConfig::new(GoogleProvider::metadata().default_model);
        GoogleProvider::from_env(model).expect("Failed to initialize Google provider")
    }
}

impl GoogleProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("GOOGLE_API_KEY")?;
        let host: String = config
            .get("GOOGLE_HOST")
            .unwrap_or_else(|_| GOOGLE_API_HOST.to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        Ok(Self {
            client,
            host,
            api_key,
            model,
        })
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        let base_url = Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;

        let url = base_url
            .join(&format!(
                "v1beta/models/{}:generateContent?key={}",
                self.model.model_name, self.api_key
            ))
            .map_err(|e| {
                ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
            })?;

        let max_retries = 5;
        let mut retries = 0;
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use rand::rngs::OsRng;
        let mut rng = StdRng::from_seed(OsRng::default().gen());

        loop {
            let response = self
                .client
                .post(url.clone()) // Clone the URL for each retry
                .header("CONTENT_TYPE", "application/json")
                .json(&payload)
                .send()
                .await;

            match response {
                Ok(res) => {
                    if res.status() == StatusCode::TOO_MANY_REQUESTS {
                        retries += 1;
                        if retries > max_retries {
                            return Err(ProviderError::RateLimitExceeded(
                                "Max retries exceeded".to_string(),
                            ));
                        }

                        let delay = 2u64.pow(retries);
                        let jitter: u64 = rng.gen_range(0..1000); // Up to 1 second of jitter, in milliseconds
                        let total_delay = Duration::from_secs(delay) + Duration::from_millis(jitter);

                        println!("Rate limit hit. Retrying in {:?}", total_delay);
                        tokio::time::sleep(total_delay).await;
                        continue;
                    } else {
                        // Successful response or other non-rate-limit error
                        return handle_response_google_compat(res).await;
                    }
                }
                Err(err) => {
                    return Err(ProviderError::RequestFailed(format!("Request failed: {}", err)));
                }
            }
        }
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "google",
            "Google Gemini",
            "Gemini models from Google AI",
            GOOGLE_DEFAULT_MODEL,
            GOOGLE_KNOWN_MODELS.iter().map(|&s| s.to_string()).collect(),
            GOOGLE_DOC_URL,
            vec![
                ConfigKey::new("GOOGLE_API_KEY", true, true, None),
                ConfigKey::new("GOOGLE_HOST", false, false, Some(GOOGLE_API_HOST)),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
async fn complete(
    &self,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<(Message, ProviderUsage), ProviderError> {
    let payload = create_request(&self.model, system, messages, tools)?;

    // Make request
    let response = self.post(payload.clone()).await;

    match response {
        Ok(response) => {
            // Parse response
            let message = response_to_message(unescape_json_values(&response))?;
            let usage = get_usage(&response)?;
            let model = match response.get("modelVersion") {
                Some(model_version) => model_version.as_str().unwrap_or_default().to_string(),
                None => self.model.model_name.clone(),
            };
            emit_debug_trace(self, &payload, &response, &usage);
            let provider_usage = ProviderUsage::new(model, usage);
            Ok((message, provider_usage))
        }
        Err(err) => {
            match err {
                ProviderError::RateLimitExceeded(msg) => {
                    println!("Rate limit exceeded: {}", msg);
                    // Return a different error or take other action
                    Err(ProviderError::Other("Gemini rate limit exceeded. Please try again later.".to_string()))
                }
                _ => {
                    // Re-raise other errors
                    Err(err)
                }
            }
        }
    }
}
}
