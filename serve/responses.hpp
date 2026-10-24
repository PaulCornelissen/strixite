#pragma once

// The OpenAI Responses request surface (/v1/responses), the wire Codex drives: validated here and normalized into
// the form the existing Chat pipeline runs (render_chat -> tokenize -> engine) - no second inference path. HTTP
// wiring in serve/http_server.cpp; the typed SSE state machine in serve/responses_stream.* (streaming) and
// responses_body.* (the non-streaming serializer). This file is host-testable: JSON in, ChatRequest + tool
// registry out, no model, no tokenizer, no GPU.
//
// Stateless by contract: store is accepted only as false (no response store, no previous_response_id). Raw
// reasoning is the model's own (reasoning_text): encrypted_content is reported null, a summary is never invented.

#include "serve/openai.hpp"

#include <string>

namespace strix {

// A validated Responses request.
//   chat: what generation consumes - messages (the input items in chat form: message items as text messages,
//     system/developer roles kept in order; tools nested in the Chat schema form; stream; model = the served id;
//     sampling; max_tokens). Null chat.tools = the generator sees no tools (none given, or tool_choice "none").
//   tool_registry: a JSON array, one entry per model-callable tool in request order, each
//     {external, kind, namespace}: the flat name the model emits (a namespaced tool as "namespace/name"), its
//     output item kind ("function" today; custom/namespace mapping: STORY-05), and the namespace to restore on
//     output ("" for top-level tools). Hosted tool kinds (web_search) are not executed here, are dropped from the
//     generator's view, and have no registry entry.
//   tool_choice: "auto" (default) | "none" - a forced choice ("required", named tool) is not served.
//   reasoning_effort: the validated effort ("" = the request named none), mapped onto the template's thinking in
//     parsing (none -> thinking off; minimal/low -> "low"; medium -> "medium"; high/xhigh -> "xhigh").
struct ResponsesRequest {
    ChatRequest chat;
    json::Value tool_registry = json::Value::array();
    std::string tool_choice = "auto";
    std::string reasoning_effort;
};

// Validates a Responses request body (parsed JSON) and normalizes it. served_model_id: the model this server
// actually runs - a request naming any other is a 400 (unlike the Chat route, which serves any name it is asked,
// this one never claims to be a model it is not). Every field is deliberate: benign client metadata parses and
// goes unused; unsupported semantics, unrecognized input item kinds and unknown fields are explicit 400s naming
// the field - never silently dropped, never silently served. Throws ApiError (400, the path in `param`).
ResponsesRequest parse_responses_request(const json::Value &body, const SamplingParams &defaults,
                                          const std::string &served_model_id);

}  // namespace strix
