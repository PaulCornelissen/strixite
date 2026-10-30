#include "serve/responses_body.hpp"

#include <stdexcept>

namespace strix {
namespace {

[[noreturn]] void fail500(const std::string &msg) {
    throw ApiError(500, msg, {}, "server_error");
}

}  // namespace

json::Value responses_body(const ResponsesRequest &rr, const std::string &resp_id, int64_t created,
                           const std::string &reasoning, const std::string &content,
                           const std::vector<ToolCallOut> &calls, const GenerationResult &res) {
    if (res.finish_reason == "error") fail500("generation failed: " + res.error);
    if (res.finish_reason == "cancelled")
        fail500("the request was cancelled before an answer (the client went away)");

    if (rr.tool_choice == "none" && !calls.empty())
        fail500("the model emitted a tool call despite tool_choice:none");
    if (!rr.parallel_tool_calls && calls.size() > 1)
        fail500("the model emitted multiple calls despite parallel_tool_calls:false");
    json::Value out = json::Value::array();
    size_t idx = 0;
    const auto item_id = [&](const char *kind) {
        std::string s = resp_id + "_" + kind + "_" + std::to_string(idx);
        ++idx;
        return s;
    };

    if (!reasoning.empty()) {
        json::Value item = json::Value::object();
        item.set("id", json::Value::string(item_id("rs")));
        item.set("type", json::Value::string("reasoning"));
        item.set("summary", json::Value::array());  // never invented
        json::Value parts = json::Value::array();
        json::Value part = json::Value::object();
        part.set("type", json::Value::string("reasoning_text"));  // the model's raw thinking
        part.set("text", json::Value::string(reasoning));
        parts.push(std::move(part));
        item.set("content", std::move(parts));
        item.set("encrypted_content", json::Value());  // null: no key, no cloud vault - disclosed
        out.push(std::move(item));
    }
    if (!content.empty()) {
        json::Value item = json::Value::object();
        item.set("id", json::Value::string(item_id("msg")));
        item.set("type", json::Value::string("message"));
        item.set("role", json::Value::string("assistant"));
        item.set("status", json::Value::string("completed"));
        json::Value parts = json::Value::array();
        json::Value part = json::Value::object();
        part.set("type", json::Value::string("output_text"));
        part.set("text", json::Value::string(content));
        part.set("annotations", json::Value::array());
        parts.push(std::move(part));
        item.set("content", std::move(parts));
        out.push(std::move(item));
    }
    for (const ToolCallOut &call : calls) {
        const json::Value *entry = nullptr;
        for (const auto &e : rr.tool_registry.as_array("registry"))
            if (e.find("external")->as_string("external") == call.name) {
                entry = &e;
                break;
            }
        // The model called a tool this request never offered: the call is meaningless to the client (its own
        // registry cannot match it). Fail the response - publishing it could be acted on blindly.
        if (entry == nullptr)
            fail500("the model emitted a call to tool '" + call.name + "' that the request did not offer");
        json::Value item = json::Value::object();
        if (entry->find("kind")->as_string("kind") == "custom") {
            const json::Value *input = call.arguments.find("input");
            if (!input || !input->is_string())
                fail500("the custom tool call '" + call.name + "' carries no string 'input'");
            item.set("id", json::Value::string(item_id("ctc")));
            item.set("type", json::Value::string("custom_tool_call"));
            item.set("status", json::Value::string("completed"));
            item.set("call_id", json::Value::string(call.id));  // the parser's id, distinct from the item id
            item.set("name", json::Value::string(entry->find("external")->as_string("external")));
            item.set("input", json::Value::string(input->as_string("input")));  // byte-for-byte
        } else {
            item.set("id", json::Value::string(item_id("fc")));
            item.set("type", json::Value::string("function_call"));
            item.set("status", json::Value::string("completed"));
            item.set("call_id", json::Value::string(call.id));
            // A namespaced child reports through its namespace + plain name; a plain tool as-is.
            const std::string &ns = entry->find("namespace")->as_string("namespace");
            const std::string &ext = entry->find("external")->as_string("external");
            item.set("name", json::Value::string(ns.empty() ? ext : ext.substr(ns.size() + 1)));
            if (!ns.empty()) item.set("namespace", json::Value::string(ns));
            item.set("arguments", json::Value::string(call.arguments.dump()));  // JSON text, as the API ships it
        }
        out.push(std::move(item));
    }
    if (out.as_array("output").empty())
        fail500("generation produced no output items (an empty turn is not a success to report)");

    json::Value body = json::Value::object();
    body.set("id", json::Value::string(resp_id));
    body.set("object", json::Value::string("response"));
    body.set("created_at", json::Value::integer(created));
    body.set("model", json::Value::string(rr.chat.model));  // the served id (parse validated it)
    const bool capped = res.finish_reason == "length";
    body.set("status", json::Value::string(capped ? "incomplete" : "completed"));
    if (capped) {
        json::Value d = json::Value::object();
        d.set("reason", json::Value::string("max_output_tokens"));
        body.set("incomplete_details", std::move(d));
    }
    body.set("error", json::Value());
    body.set("output", std::move(out));
    body.set("output_text", json::Value::string(content));
    body.set("parallel_tool_calls", json::Value::boolean(rr.parallel_tool_calls));
    body.set("store", json::Value::boolean(false));  // stateless by contract (validated at parse)

    json::Value usage = json::Value::object();
    usage.set("input_tokens", json::Value::integer(res.prompt_tokens));
    json::Value in_d = json::Value::object();
    in_d.set("cached_tokens", json::Value::integer(res.cached_tokens));
    usage.set("input_tokens_details", std::move(in_d));
    usage.set("output_tokens", json::Value::integer(res.completion_tokens));
    json::Value out_d = json::Value::object();
    out_d.set("reasoning_tokens", json::Value::integer(res.reasoning_tokens));
    usage.set("output_tokens_details", std::move(out_d));
    usage.set("total_tokens", json::Value::integer(res.prompt_tokens + res.completion_tokens));
    body.set("usage", std::move(usage));
    return body;
}

}  // namespace strix
