//! Anthropic Messages API compatibility for HTTP routers.
//!
//! The gateway workers expose OpenAI Chat Completions. This module translates
//! Anthropic `/v1/messages` requests and responses at the server boundary so
//! regular and PD routers can reuse the existing `route_chat` implementation.

use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    pin::Pin,
};

use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, response::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{stream, Stream, StreamExt};
use serde_json::{json, Map, Value};

use crate::protocols::chat::ChatCompletionRequest;

const MAX_RESPONSE_BODY_SIZE: usize = 32 * 1024 * 1024;

pub struct ConvertedRequest {
    pub chat: ChatCompletionRequest,
    pub response_config: ResponseConfig,
}

#[derive(Clone)]
pub struct ResponseConfig {
    pub model: String,
    pub stream: bool,
    pub include_thinking: bool,
    pub raw_thinking_fallback: bool,
}

#[derive(Debug)]
pub struct AnthropicError {
    status: StatusCode,
    error_type: &'static str,
    message: String,
}

impl AnthropicError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "api_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "type": "error",
                "error": {
                    "type": self.error_type,
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

pub fn convert_request(body: Value) -> Result<ConvertedRequest, AnthropicError> {
    let request = body
        .as_object()
        .ok_or_else(|| AnthropicError::invalid("request body must be a JSON object"))?;

    let model = required_string(request, "model")?.to_string();
    let max_tokens = request
        .get("max_tokens")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| AnthropicError::invalid("max_tokens must be a positive integer"))?;
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| AnthropicError::invalid("messages must be an array"))?;
    if messages.is_empty() {
        return Err(AnthropicError::invalid("messages must not be empty"));
    }

    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let thinking_type = request
        .get("thinking")
        .and_then(Value::as_object)
        .and_then(|thinking| thinking.get("type"))
        .and_then(Value::as_str);
    // Kimi can return reasoning_content even when the Anthropic request does
    // not include a thinking configuration. Preserve that content unless the
    // client explicitly disables it.
    let include_thinking = thinking_type != Some("disabled");
    let raw_thinking_fallback = include_thinking
        && (thinking_type == Some("enabled") || model.to_ascii_lowercase().contains("kimi2.7"));

    let mut openai = Map::new();
    openai.insert("model".to_string(), Value::String(model.clone()));
    openai.insert("max_tokens".to_string(), json!(max_tokens));
    openai.insert("stream".to_string(), Value::Bool(stream));
    if stream {
        openai.insert("stream_options".to_string(), json!({"include_usage": true}));
    }

    let mut openai_messages = Vec::new();
    if let Some(system) = request.get("system") {
        let content = content_to_text(system, "system")?;
        if !content.is_empty() {
            openai_messages.push(json!({"role": "system", "content": content}));
        }
    }
    openai_messages.extend(convert_messages(messages)?);
    openai.insert("messages".to_string(), Value::Array(openai_messages));

    copy_number(request, &mut openai, "temperature");
    copy_number(request, &mut openai, "top_p");
    copy_number(request, &mut openai, "top_k");

    if let Some(stop_sequences) = request.get("stop_sequences") {
        if !stop_sequences.is_array() {
            return Err(AnthropicError::invalid(
                "stop_sequences must be an array of strings",
            ));
        }
        openai.insert("stop".to_string(), stop_sequences.clone());
    }

    if let Some(user_id) = request
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("user_id"))
        .and_then(Value::as_str)
    {
        openai.insert("user".to_string(), Value::String(user_id.to_string()));
    }

    if let Some(tools) = request.get("tools") {
        openai.insert("tools".to_string(), convert_tools(tools)?);
    }

    if let Some(tool_choice) = request.get("tool_choice") {
        let (choice, parallel) = convert_tool_choice(tool_choice)?;
        openai.insert("tool_choice".to_string(), choice);
        if let Some(parallel) = parallel {
            openai.insert("parallel_tool_calls".to_string(), Value::Bool(parallel));
        }
    }

    if request.get("thinking").is_some() {
        openai.insert(
            "chat_template_kwargs".to_string(),
            json!({"enable_thinking": include_thinking}),
        );
    }

    let chat = serde_json::from_value(Value::Object(openai)).map_err(|error| {
        AnthropicError::invalid(format!(
            "request cannot be converted to Chat Completions: {error}"
        ))
    })?;

    Ok(ConvertedRequest {
        chat,
        response_config: ResponseConfig {
            model,
            stream,
            include_thinking,
            raw_thinking_fallback,
        },
    })
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, AnthropicError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AnthropicError::invalid(format!("{field} must be a non-empty string")))
}

fn copy_number(source: &Map<String, Value>, target: &mut Map<String, Value>, field: &str) {
    if let Some(value) = source.get(field).filter(|value| value.is_number()) {
        target.insert(field.to_string(), value.clone());
    }
}

fn convert_messages(messages: &[Value]) -> Result<Vec<Value>, AnthropicError> {
    let mut result = Vec::new();

    for message in messages {
        let message = message
            .as_object()
            .ok_or_else(|| AnthropicError::invalid("each message must be an object"))?;
        let role = required_string(message, "role")?;
        let content = message
            .get("content")
            .ok_or_else(|| AnthropicError::invalid("message content is required"))?;

        match role {
            "assistant" => result.push(convert_assistant_message(content)?),
            "user" => result.extend(convert_user_message(content)?),
            _ => {
                return Err(AnthropicError::invalid(
                    "Anthropic message role must be user or assistant",
                ));
            }
        }
    }

    Ok(result)
}

fn convert_assistant_message(content: &Value) -> Result<Value, AnthropicError> {
    if let Some(text) = content.as_str() {
        return Ok(json!({"role": "assistant", "content": text}));
    }

    let blocks = content.as_array().ok_or_else(|| {
        AnthropicError::invalid("assistant content must be a string or content block array")
    })?;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        let block = block
            .as_object()
            .ok_or_else(|| AnthropicError::invalid("content block must be an object"))?;
        match block.get("type").and_then(Value::as_str) {
            Some("text") => append_text(&mut text, block.get("text").and_then(Value::as_str)),
            Some("thinking") => append_text(
                &mut reasoning,
                block.get("thinking").and_then(Value::as_str),
            ),
            Some("tool_use") => {
                let id = required_string(block, "id")?;
                let name = required_string(block, "name")?;
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input).map_err(|error| {
                            AnthropicError::invalid(format!(
                                "tool_use input is not serializable: {error}"
                            ))
                        })?,
                    }
                }));
            }
            Some("redacted_thinking") => {}
            Some(other) => {
                return Err(AnthropicError::invalid(format!(
                    "unsupported assistant content block type: {other}"
                )));
            }
            None => return Err(AnthropicError::invalid("content block type is required")),
        }
    }

    let mut message = Map::new();
    message.insert("role".to_string(), Value::String("assistant".to_string()));
    message.insert(
        "content".to_string(),
        if text.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        },
    );
    if !reasoning.is_empty() {
        message.insert("reasoning_content".to_string(), Value::String(reasoning));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    Ok(Value::Object(message))
}

fn convert_user_message(content: &Value) -> Result<Vec<Value>, AnthropicError> {
    if let Some(text) = content.as_str() {
        return Ok(vec![json!({"role": "user", "content": text})]);
    }

    let blocks = content.as_array().ok_or_else(|| {
        AnthropicError::invalid("user content must be a string or content block array")
    })?;
    let mut result = Vec::new();
    let mut parts = Vec::new();

    for block in blocks {
        let block = block
            .as_object()
            .ok_or_else(|| AnthropicError::invalid("content block must be an object"))?;
        match block.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(json!({
                "type": "text",
                "text": block.get("text").and_then(Value::as_str).unwrap_or_default(),
            })),
            Some("image") => parts.push(convert_image_block(block)?),
            Some("tool_result") => {
                flush_user_parts(&mut result, &mut parts);
                let tool_call_id = required_string(block, "tool_use_id")?;
                let tool_content = block
                    .get("content")
                    .map(|value| content_to_text(value, "tool_result"))
                    .transpose()?
                    .unwrap_or_default();
                result.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                }));
            }
            Some(other) => {
                return Err(AnthropicError::invalid(format!(
                    "unsupported user content block type: {other}"
                )));
            }
            None => return Err(AnthropicError::invalid("content block type is required")),
        }
    }
    flush_user_parts(&mut result, &mut parts);

    if result.is_empty() {
        result.push(json!({"role": "user", "content": ""}));
    }
    Ok(result)
}

fn flush_user_parts(result: &mut Vec<Value>, parts: &mut Vec<Value>) {
    if !parts.is_empty() {
        result.push(json!({
            "role": "user",
            "content": Value::Array(std::mem::take(parts)),
        }));
    }
}

fn convert_image_block(block: &Map<String, Value>) -> Result<Value, AnthropicError> {
    let source = block
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| AnthropicError::invalid("image source must be an object"))?;
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = required_string(source, "media_type")?;
            let data = required_string(source, "data")?;
            format!("data:{media_type};base64,{data}")
        }
        Some("url") => required_string(source, "url")?.to_string(),
        Some(other) => {
            return Err(AnthropicError::invalid(format!(
                "unsupported image source type: {other}"
            )));
        }
        None => return Err(AnthropicError::invalid("image source type is required")),
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

fn content_to_text(content: &Value, field: &str) -> Result<String, AnthropicError> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_string());
    }
    let blocks = content.as_array().ok_or_else(|| {
        AnthropicError::invalid(format!("{field} content must be a string or block array"))
    })?;
    let mut text = String::new();
    for block in blocks {
        let block = block
            .as_object()
            .ok_or_else(|| AnthropicError::invalid(format!("{field} block must be an object")))?;
        if block.get("type").and_then(Value::as_str) != Some("text") {
            return Err(AnthropicError::invalid(format!(
                "{field} only supports text blocks"
            )));
        }
        append_text(&mut text, block.get("text").and_then(Value::as_str));
    }
    Ok(text)
}

fn append_text(target: &mut String, text: Option<&str>) {
    if let Some(text) = text {
        if !target.is_empty() && !text.is_empty() {
            target.push('\n');
        }
        target.push_str(text);
    }
}

fn convert_tools(tools: &Value) -> Result<Value, AnthropicError> {
    let tools = tools
        .as_array()
        .ok_or_else(|| AnthropicError::invalid("tools must be an array"))?;
    let mut converted = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool = tool
            .as_object()
            .ok_or_else(|| AnthropicError::invalid("each tool must be an object"))?;
        let name = required_string(tool, "name")?;
        let input_schema = tool
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"}));
        converted.push(json!({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description").cloned().unwrap_or(Value::Null),
                "parameters": input_schema,
            }
        }));
    }
    Ok(Value::Array(converted))
}

fn convert_tool_choice(tool_choice: &Value) -> Result<(Value, Option<bool>), AnthropicError> {
    let tool_choice = tool_choice
        .as_object()
        .ok_or_else(|| AnthropicError::invalid("tool_choice must be an object"))?;
    let choice_type = required_string(tool_choice, "type")?;
    let parallel = tool_choice
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .map(|disabled| !disabled);
    let choice = match choice_type {
        "auto" => Value::String("auto".to_string()),
        "any" => Value::String("required".to_string()),
        "none" => Value::String("none".to_string()),
        "tool" => json!({
            "type": "function",
            "function": {"name": required_string(tool_choice, "name")?},
        }),
        other => {
            return Err(AnthropicError::invalid(format!(
                "unsupported tool_choice type: {other}"
            )));
        }
    };
    Ok((choice, parallel))
}

pub async fn transform_response(response: Response, config: ResponseConfig) -> Response {
    let (parts, body) = response.into_parts();
    if !parts.status.is_success() {
        return transform_error_response(parts, body).await;
    }

    if config.stream {
        transform_streaming_response(parts, body, config)
    } else {
        transform_json_response(parts, body, config).await
    }
}

async fn transform_json_response(mut parts: Parts, body: Body, config: ResponseConfig) -> Response {
    let bytes = match to_bytes(body, MAX_RESPONSE_BODY_SIZE).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return AnthropicError::internal(format!(
                "failed to read Chat Completions response: {error}"
            ))
            .into_response();
        }
    };
    let openai: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return AnthropicError::internal(format!("invalid Chat Completions response: {error}"))
                .into_response();
        }
    };
    let anthropic = match convert_non_stream_value(&openai, &config) {
        Ok(value) => value,
        Err(error) => return error.into_response(),
    };
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::CONTENT_ENCODING);
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, Body::from(anthropic.to_string()))
}

fn convert_non_stream_value(
    openai: &Value,
    config: &ResponseConfig,
) -> Result<Value, AnthropicError> {
    let choice = openai
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| AnthropicError::internal("Chat Completions response has no choices"))?;
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| AnthropicError::internal("Chat Completions response has no message"))?;

    let mut content = Vec::new();
    let mut text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let reasoning = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|reasoning| !reasoning.is_empty());
    if config.include_thinking {
        if let Some(reasoning) = reasoning {
            content.push(thinking_block(reasoning));
        } else if config.raw_thinking_fallback {
            if let Some((thinking, answer)) = split_raw_thinking(text) {
                if !thinking.is_empty() {
                    content.push(thinking_block(thinking));
                }
                text = answer;
            } else if choice.get("finish_reason").and_then(Value::as_str) == Some("length") {
                let thinking = normalize_raw_thinking(text);
                if !thinking.is_empty() {
                    content.push(thinking_block(thinking));
                    text = "";
                }
            }
        }
    }
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            let function = tool_call
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| AnthropicError::internal("tool call has no function"))?;
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|arguments| serde_json::from_str(arguments).ok())
                .unwrap_or_else(|| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": tool_call.get("id").and_then(Value::as_str).unwrap_or("toolu_unknown"),
                "name": function.get("name").and_then(Value::as_str).unwrap_or("unknown"),
                "input": arguments,
            }));
        }
    }

    let usage = openai.get("usage").cloned().unwrap_or_else(|| json!({}));
    Ok(json!({
        "id": anthropic_message_id(openai.get("id").and_then(Value::as_str)),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": openai
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&config.model),
        "stop_reason": map_stop_reason(choice.get("finish_reason").and_then(Value::as_str)),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            "output_tokens": usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
        }
    }))
}

fn thinking_block(thinking: &str) -> Value {
    json!({
        "type": "thinking",
        "thinking": thinking,
        "signature": "",
    })
}

fn split_raw_thinking(text: &str) -> Option<(&str, &str)> {
    let (thinking, answer) = text.split_once("</think>")?;
    Some((normalize_raw_thinking(thinking), answer.trim_start()))
}

fn normalize_raw_thinking(text: &str) -> &str {
    let text = text.trim_start();
    text.strip_prefix("<think>").unwrap_or(text).trim()
}

async fn transform_error_response(mut parts: Parts, body: Body) -> Response {
    let bytes = to_bytes(body, MAX_RESPONSE_BODY_SIZE)
        .await
        .unwrap_or_default();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let message = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .unwrap_or("upstream request failed");
    let error_type = error_type_for_status(parts.status);
    let body = json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::CONTENT_ENCODING);
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, Body::from(body.to_string()))
}

fn error_type_for_status(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        status if status.as_u16() == 529 || status == StatusCode::SERVICE_UNAVAILABLE => {
            "overloaded_error"
        }
        _ => "api_error",
    }
}

fn anthropic_message_id(openai_id: Option<&str>) -> String {
    match openai_id {
        Some(id) if id.starts_with("msg_") => id.to_string(),
        Some(id) => format!("msg_{}", id.trim_start_matches("chatcmpl-")),
        None => "msg_unknown".to_string(),
    }
}

fn map_stop_reason(reason: Option<&str>) -> Value {
    match reason {
        Some("length") => Value::String("max_tokens".to_string()),
        Some("tool_calls") | Some("function_call") => Value::String("tool_use".to_string()),
        Some("stop") => Value::String("end_turn".to_string()),
        Some("content_filter") => Value::String("refusal".to_string()),
        Some(other) => Value::String(other.to_string()),
        None => Value::Null,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BlockKind {
    Thinking,
    Text,
    Tool(u64),
}

struct AnthropicStreamConverter {
    config: ResponseConfig,
    started: bool,
    finished: bool,
    message_id: String,
    model: String,
    current_block: Option<BlockKind>,
    current_content_index: u64,
    next_content_index: u64,
    finish_reason: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    tool_ids: HashMap<u64, String>,
    tool_names: HashMap<u64, String>,
    structured_thinking_seen: bool,
    raw_thinking_closed: bool,
    raw_thinking_buffer: String,
}

impl AnthropicStreamConverter {
    fn new(config: ResponseConfig) -> Self {
        Self {
            model: config.model.clone(),
            config,
            started: false,
            finished: false,
            message_id: "msg_unknown".to_string(),
            current_block: None,
            current_content_index: 0,
            next_content_index: 0,
            finish_reason: None,
            input_tokens: 0,
            output_tokens: 0,
            tool_ids: HashMap::new(),
            tool_names: HashMap::new(),
            structured_thinking_seen: false,
            raw_thinking_closed: false,
            raw_thinking_buffer: String::new(),
        }
    }

    fn push_data(&mut self, data: &str) -> Vec<String> {
        if data.trim() == "[DONE]" {
            return self.finish();
        }

        let chunk: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(error) => {
                return vec![format_sse(
                    "error",
                    json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("invalid upstream SSE event: {error}"),
                        }
                    }),
                )];
            }
        };

        if let Some(id) = chunk.get("id").and_then(Value::as_str) {
            self.message_id = anthropic_message_id(Some(id));
        }
        if let Some(model) = chunk.get("model").and_then(Value::as_str) {
            self.model = model.to_string();
        }
        self.update_usage(chunk.get("usage"));

        let mut events = Vec::new();
        self.ensure_message_start(&mut events);
        if let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                if self.config.include_thinking {
                    if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str)
                    {
                        if !reasoning.is_empty() {
                            self.structured_thinking_seen = true;
                            self.flush_raw_thinking_buffer(&mut events);
                            self.push_thinking(reasoning, &mut events);
                        }
                    }
                }
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    if !text.is_empty() {
                        self.push_text(text, &mut events);
                    }
                }
                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tool_call in tool_calls {
                        self.push_tool_call(tool_call, &mut events);
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(reason.to_string());
            }
        }
        events
    }

    fn push_thinking(&mut self, thinking: &str, events: &mut Vec<String>) {
        self.ensure_block(BlockKind::Thinking, None, None, events);
        events.push(format_sse(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": self.current_content_index,
                "delta": {"type": "thinking_delta", "thinking": thinking},
            }),
        ));
    }

    fn push_text(&mut self, text: &str, events: &mut Vec<String>) {
        if self.config.raw_thinking_fallback
            && !self.structured_thinking_seen
            && !self.raw_thinking_closed
        {
            self.raw_thinking_buffer.push_str(text);
            if let Some(end) = self.raw_thinking_buffer.find("</think>") {
                let buffered = std::mem::take(&mut self.raw_thinking_buffer);
                let (thinking, answer_with_tag) = buffered.split_at(end);
                let answer = answer_with_tag["</think>".len()..].trim_start();
                let thinking = normalize_raw_thinking(thinking);
                if !thinking.is_empty() {
                    self.push_thinking(thinking, events);
                }
                self.raw_thinking_closed = true;
                if !answer.is_empty() {
                    self.push_text_delta(answer, events);
                }
            }
            return;
        }
        self.push_text_delta(text, events);
    }

    fn push_text_delta(&mut self, text: &str, events: &mut Vec<String>) {
        self.ensure_block(BlockKind::Text, None, None, events);
        events.push(format_sse(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": self.current_content_index,
                "delta": {"type": "text_delta", "text": text},
            }),
        ));
    }

    fn flush_raw_thinking_buffer(&mut self, events: &mut Vec<String>) {
        if self.raw_thinking_buffer.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.raw_thinking_buffer);
        if self.config.raw_thinking_fallback
            && !self.structured_thinking_seen
            && !self.raw_thinking_closed
            && self.finish_reason.as_deref() == Some("length")
        {
            let thinking = normalize_raw_thinking(&text);
            if !thinking.is_empty() {
                self.push_thinking(thinking, events);
                return;
            }
        }
        self.push_text_delta(&text, events);
    }

    fn push_tool_call(&mut self, tool_call: &Value, events: &mut Vec<String>) {
        let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0);
        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            self.tool_ids.insert(index, id.to_string());
        }
        let function = tool_call.get("function").and_then(Value::as_object);
        if let Some(name) = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
        {
            self.tool_names.insert(index, name.to_string());
        }
        let id = self
            .tool_ids
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("toolu_{index}"));
        let name = self
            .tool_names
            .get(&index)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        self.ensure_block(BlockKind::Tool(index), Some(&id), Some(&name), events);

        if let Some(arguments) = function
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
        {
            if !arguments.is_empty() {
                events.push(format_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.current_content_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments,
                        },
                    }),
                ));
            }
        }
    }

    fn update_usage(&mut self, usage: Option<&Value>) {
        if let Some(usage) = usage {
            if let Some(tokens) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                self.input_tokens = tokens;
            }
            if let Some(tokens) = usage.get("completion_tokens").and_then(Value::as_u64) {
                self.output_tokens = tokens;
            }
        }
    }

    fn ensure_message_start(&mut self, events: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        events.push(format_sse(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            }),
        ));
    }

    fn ensure_block(
        &mut self,
        kind: BlockKind,
        tool_id: Option<&str>,
        tool_name: Option<&str>,
        events: &mut Vec<String>,
    ) {
        if self.current_block.as_ref() == Some(&kind) {
            return;
        }
        self.close_block(events);
        self.current_content_index = self.next_content_index;
        self.next_content_index += 1;
        let content_block = match kind {
            BlockKind::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            BlockKind::Text => json!({"type": "text", "text": ""}),
            BlockKind::Tool(_) => json!({
                "type": "tool_use",
                "id": tool_id.unwrap_or("toolu_unknown"),
                "name": tool_name.unwrap_or("unknown"),
                "input": {},
            }),
        };
        self.current_block = Some(kind);
        events.push(format_sse(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": self.current_content_index,
                "content_block": content_block,
            }),
        ));
    }

    fn close_block(&mut self, events: &mut Vec<String>) {
        if let Some(kind) = self.current_block.take() {
            if kind == BlockKind::Thinking {
                // Kimi's OpenAI response has no Anthropic cryptographic
                // signature. Emit an explicitly empty compatibility value
                // instead of inventing a signature.
                events.push(format_sse(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.current_content_index,
                        "delta": {"type": "signature_delta", "signature": ""},
                    }),
                ));
            }
            events.push(format_sse(
                "content_block_stop",
                json!({
                    "type": "content_block_stop",
                    "index": self.current_content_index,
                }),
            ));
        }
    }

    fn finish(&mut self) -> Vec<String> {
        if self.finished {
            return Vec::new();
        }
        let mut events = Vec::new();
        self.ensure_message_start(&mut events);
        self.flush_raw_thinking_buffer(&mut events);
        self.close_block(&mut events);
        events.push(format_sse(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": map_stop_reason(self.finish_reason.as_deref()),
                    "stop_sequence": Value::Null,
                },
                "usage": {"output_tokens": self.output_tokens},
            }),
        ));
        events.push(format_sse("message_stop", json!({"type": "message_stop"})));
        self.finished = true;
        events
    }
}

type UpstreamByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, axum::Error>> + Send + 'static>>;

struct StreamingState {
    upstream: UpstreamByteStream,
    buffer: Vec<u8>,
    pending: VecDeque<Bytes>,
    converter: AnthropicStreamConverter,
    upstream_done: bool,
}

fn transform_streaming_response(mut parts: Parts, body: Body, config: ResponseConfig) -> Response {
    let state = StreamingState {
        upstream: Box::pin(body.into_data_stream()),
        buffer: Vec::new(),
        pending: VecDeque::new(),
        converter: AnthropicStreamConverter::new(config),
        upstream_done: false,
    };

    let output = stream::unfold(state, |mut state| async move {
        loop {
            if let Some(bytes) = state.pending.pop_front() {
                return Some((Ok::<Bytes, Infallible>(bytes), state));
            }
            if let Some(event) = take_sse_event(&mut state.buffer) {
                if let Some(data) = extract_sse_data(&event) {
                    state.pending.extend(
                        state
                            .converter
                            .push_data(&data)
                            .into_iter()
                            .map(Bytes::from),
                    );
                    continue;
                }
            }
            if state.upstream_done {
                state
                    .pending
                    .extend(state.converter.finish().into_iter().map(Bytes::from));
                if state.pending.is_empty() {
                    return None;
                }
                continue;
            }
            match state.upstream.next().await {
                Some(Ok(bytes)) => state.buffer.extend_from_slice(&bytes),
                Some(Err(error)) => {
                    state.pending.push_back(Bytes::from(format_sse(
                        "error",
                        json!({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": format!("upstream stream failed: {error}"),
                            },
                        }),
                    )));
                    state.upstream_done = true;
                }
                None => {
                    if !state.buffer.is_empty() {
                        state.buffer.extend_from_slice(b"\n\n");
                    }
                    state.upstream_done = true;
                }
            }
        }
    });

    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::CONTENT_ENCODING);
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("text/event-stream"),
    );
    parts.headers.insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-cache"),
    );
    Response::from_parts(parts, Body::from_stream(output))
}

fn take_sse_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let lf = buffer.windows(2).position(|window| window == b"\n\n");
    let crlf = buffer.windows(4).position(|window| window == b"\r\n\r\n");
    let (position, delimiter_len) = match (lf, crlf) {
        (Some(lf), Some(crlf)) if lf <= crlf => (lf, 2),
        (Some(_), Some(crlf)) => (crlf, 4),
        (Some(lf), None) => (lf, 2),
        (None, Some(crlf)) => (crlf, 4),
        (None, None) => return None,
    };
    let event = buffer[..position].to_vec();
    buffer.drain(..position + delimiter_len);
    Some(event)
}

fn extract_sse_data(event: &[u8]) -> Option<String> {
    let event = String::from_utf8_lossy(event);
    let data: Vec<&str> = event
        .lines()
        .filter_map(|line| line.trim_end_matches('\r').strip_prefix("data:"))
        .map(str::trim_start)
        .collect();
    (!data.is_empty()).then(|| data.join("\n"))
}

fn format_sse(event: &str, data: Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_anthropic_request_with_system_tools_and_tool_result() {
        let converted = convert_request(json!({
            "model": "kimi-2.7",
            "max_tokens": 128,
            "thinking": {"type": "enabled", "budget_tokens": 64},
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "weather",
                        "input": {"city": "Shanghai"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "sunny"
                    }]
                }
            ],
            "tools": [{
                "name": "weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }
            }],
            "tool_choice": {"type": "tool", "name": "weather"},
            "stream": false
        }))
        .unwrap();

        let value = serde_json::to_value(converted.chat).unwrap();
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(
            value["messages"][1]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(value["messages"][2]["role"], "tool");
        assert_eq!(value["tools"][0]["function"]["name"], "weather");
        assert_eq!(
            value["tool_choice"]["function"]["name"],
            Value::String("weather".to_string())
        );
        assert_eq!(
            value["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(true)
        );
        assert!(converted.response_config.include_thinking);
    }

    #[test]
    fn enables_raw_thinking_fallback_for_kimi_without_thinking_config() {
        let converted = convert_request(json!({
            "model": "Kimi2.7",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}]
        }))
        .unwrap();

        assert!(converted.response_config.include_thinking);
        assert!(converted.response_config.raw_thinking_fallback);
    }

    #[test]
    fn requests_usage_for_streaming_responses() {
        let converted = convert_request(json!({
            "model": "Kimi2.7",
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}]
        }))
        .unwrap();
        let value = serde_json::to_value(converted.chat).unwrap();

        assert_eq!(value["stream_options"]["include_usage"], Value::Bool(true));
    }

    #[test]
    fn rejects_missing_required_fields() {
        let error = convert_request(json!({"model": "kimi-2.7", "messages": []}))
            .err()
            .unwrap();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.contains("max_tokens"));
    }

    #[test]
    fn converts_non_stream_response() {
        let response = convert_non_stream_value(
            &json!({
                "id": "chatcmpl-123",
                "model": "kimi-2.7",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{\"city\":\"Shanghai\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            }),
            &ResponseConfig {
                model: "kimi-2.7".to_string(),
                stream: false,
                include_thinking: false,
                raw_thinking_fallback: false,
            },
        )
        .unwrap();

        assert_eq!(response["id"], "msg_123");
        assert_eq!(response["content"][0]["type"], "text");
        assert_eq!(response["content"][1]["type"], "tool_use");
        assert_eq!(response["stop_reason"], "tool_use");
        assert_eq!(response["usage"]["input_tokens"], 10);
        assert_eq!(response["usage"]["output_tokens"], 5);
    }

    #[test]
    fn splits_raw_kimi_thinking_in_non_stream_response() {
        let response = convert_non_stream_value(
            &json!({
                "id": "chatcmpl-raw-thinking",
                "model": "kimi-2.7",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Compute 17 times 23.</think>391"
                    },
                    "finish_reason": "stop"
                }]
            }),
            &ResponseConfig {
                model: "kimi-2.7".to_string(),
                stream: false,
                include_thinking: true,
                raw_thinking_fallback: true,
            },
        )
        .unwrap();

        assert_eq!(response["content"][0]["type"], "thinking");
        assert_eq!(response["content"][0]["thinking"], "Compute 17 times 23.");
        assert_eq!(response["content"][1]["type"], "text");
        assert_eq!(response["content"][1]["text"], "391");
    }

    #[test]
    fn keeps_truncated_raw_kimi_output_as_thinking() {
        let response = convert_non_stream_value(
            &json!({
                "id": "chatcmpl-truncated-thinking",
                "model": "kimi-2.7",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "The user wants me to reply with exactly"
                    },
                    "finish_reason": "length"
                }]
            }),
            &ResponseConfig {
                model: "kimi-2.7".to_string(),
                stream: false,
                include_thinking: true,
                raw_thinking_fallback: true,
            },
        )
        .unwrap();

        assert_eq!(response["content"].as_array().unwrap().len(), 1);
        assert_eq!(response["content"][0]["type"], "thinking");
        assert_eq!(
            response["content"][0]["thinking"],
            "The user wants me to reply with exactly"
        );
        assert_eq!(response["stop_reason"], "max_tokens");
    }

    #[test]
    fn converts_stream_events_in_anthropic_order() {
        let mut converter = AnthropicStreamConverter::new(ResponseConfig {
            model: "kimi-2.7".to_string(),
            stream: true,
            include_thinking: true,
            raw_thinking_fallback: false,
        });
        let mut events = converter.push_data(
            &json!({
                "id": "chatcmpl-123",
                "model": "kimi-2.7",
                "choices": [{"delta": {"reasoning_content": "Plan"}, "finish_reason": null}]
            })
            .to_string(),
        );
        events.extend(
            converter.push_data(
                &json!({
                    "choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]
                })
                .to_string(),
            ),
        );
        events.extend(
            converter.push_data(
                &json!({
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1}
                })
                .to_string(),
            ),
        );
        events.extend(converter.push_data("[DONE]"));
        let output = events.join("");

        let message_start = output.find("event: message_start").unwrap();
        let block_start = output.find("event: content_block_start").unwrap();
        let block_delta = output.find("event: content_block_delta").unwrap();
        let block_stop = output.find("event: content_block_stop").unwrap();
        let message_delta = output.find("event: message_delta").unwrap();
        let message_stop = output.find("event: message_stop").unwrap();
        assert!(message_start < block_start);
        assert!(block_start < block_delta);
        assert!(block_delta < block_stop);
        assert!(block_stop < message_delta);
        assert!(message_delta < message_stop);
        assert!(output.contains("\"type\":\"thinking_delta\""));
        assert!(output.contains("\"type\":\"signature_delta\""));
        assert!(output.contains("\"index\":1"));
        assert!(output.contains("\"stop_reason\":\"end_turn\""));
        assert!(output.contains("\"output_tokens\":1"));
    }

    #[test]
    fn splits_chunked_raw_kimi_thinking_in_stream_response() {
        let mut converter = AnthropicStreamConverter::new(ResponseConfig {
            model: "kimi-2.7".to_string(),
            stream: true,
            include_thinking: true,
            raw_thinking_fallback: true,
        });
        let mut events = converter.push_data(
            &json!({
                "id": "chatcmpl-raw-thinking",
                "choices": [{"delta": {"content": "Compute 17 times 23.</thi"}}]
            })
            .to_string(),
        );
        events.extend(
            converter.push_data(
                &json!({
                    "choices": [{"delta": {"content": "nk>391"}}]
                })
                .to_string(),
            ),
        );
        events.extend(
            converter.push_data(
                &json!({
                    "choices": [{"delta": {}, "finish_reason": "stop"}]
                })
                .to_string(),
            ),
        );
        events.extend(converter.push_data("[DONE]"));
        let output = events.join("");

        assert!(
            output.contains("\"type\":\"thinking_delta\",\"thinking\":\"Compute 17 times 23.\"")
        );
        assert!(output.contains("\"type\":\"text_delta\",\"text\":\"391\""));
        assert!(output.contains("\"type\":\"signature_delta\""));
        assert!(output.contains("\"index\":1"));
        assert!(!output.contains("</think>"));
    }

    #[test]
    fn keeps_truncated_streamed_kimi_output_as_thinking() {
        let mut converter = AnthropicStreamConverter::new(ResponseConfig {
            model: "kimi-2.7".to_string(),
            stream: true,
            include_thinking: true,
            raw_thinking_fallback: true,
        });
        let mut events = converter.push_data(
            &json!({
                "id": "chatcmpl-truncated-thinking",
                "choices": [{
                    "delta": {"content": "The user wants me to reply with exactly"},
                    "finish_reason": null
                }]
            })
            .to_string(),
        );
        events.extend(
            converter.push_data(
                &json!({
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 8}
                })
                .to_string(),
            ),
        );
        events.extend(converter.push_data("[DONE]"));
        let output = events.join("");

        assert!(output.contains("\"type\":\"thinking_delta\""));
        assert!(!output.contains("\"type\":\"text_delta\""));
        assert!(output.contains("\"stop_reason\":\"max_tokens\""));
        assert!(output.contains("\"output_tokens\":8"));
    }

    #[test]
    fn parses_crlf_and_lf_sse_boundaries() {
        let mut buffer = b"data: one\r\n\r\ndata: two\n\nrest".to_vec();
        assert_eq!(
            extract_sse_data(&take_sse_event(&mut buffer).unwrap()),
            Some("one".into())
        );
        assert_eq!(
            extract_sse_data(&take_sse_event(&mut buffer).unwrap()),
            Some("two".into())
        );
        assert_eq!(buffer, b"rest");
    }
}
