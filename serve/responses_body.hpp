#pragma once

// The non-streaming /v1/responses body: generation's outcome (raw reasoning, answer text, the calls the
// buffered parser completed) serialized as typed Responses output items with real usage. Statuses are exact -
// `completed` only on a normal end, `incomplete` on the output cap (with incomplete_details), a server-side
// inconsistency (a call naming a tool this request never offered, a custom call without its input) throws
// ApiError 500: the route answers a typed failure, never a body Codex could act on wrongly. Raw reasoning is
// the model's own text: summary is the empty array, encrypted_content is null (this server has no key).

#include "serve/engine.hpp"       // GenerationResult
#include "serve/openai.hpp"       // ApiError
#include "serve/output_parser.hpp"  // ToolCallOut
#include "serve/responses.hpp"    // ResponsesRequest (registry, model)

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

// One generated response, as the Responses JSON. resp_id ("resp_..."), created: unix seconds. Item ids are the
// response's (id + index); tool calls keep their own call_id (the parser's "call_..."): the two never mix.
json::Value responses_body(const ResponsesRequest &rr, const std::string &resp_id, int64_t created,
                           const std::string &reasoning, const std::string &content,
                           const std::vector<ToolCallOut> &calls, const GenerationResult &res);

}  // namespace strix
