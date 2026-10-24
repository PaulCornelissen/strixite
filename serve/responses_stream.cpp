#include "serve/responses_stream.hpp"

#include <stdexcept>

namespace strix {
namespace {

json::Value ev(const char *type) {
    json::Value v = json::Value::object();
    v.set("type", json::Value::string(type));
    return v;
}

std::string frame(json::Value ev_obj) {
    return "data: " + ev_obj.dump() + "\n\n";
}

json::Value usage_json_of(const GenerationResult &r) {
    json::Value u = json::Value::object();
    u.set("input_tokens", json::Value::integer(r.prompt_tokens));
    json::Value in_d = json::Value::object();
    in_d.set("cached_tokens", json::Value::integer(r.cached_tokens));
    u.set("input_tokens_details", std::move(in_d));
    u.set("output_tokens", json::Value::integer(r.completion_tokens));
    json::Value out_d = json::Value::object();
    out_d.set("reasoning_tokens", json::Value::integer(r.reasoning_tokens));
    u.set("output_tokens_details", std::move(out_d));
    u.set("total_tokens", json::Value::integer(r.prompt_tokens + r.completion_tokens));
    return u;
}

}  // namespace

ResponsesStream::ResponsesStream(std::string resp_id, std::string model, int64_t created, json::Value registry)
    : id_(std::move(resp_id)), model_(std::move(model)), created_(created), registry_(std::move(registry)) {}

void ResponsesStream::push_event(json::Value v) {
    v.set("sequence_number", json::Value::integer(seq_++));
    queue_ += frame(std::move(v));
}

std::string ResponsesStream::ev_created() {
    if (started_) return {};
    started_ = true;
    json::Value body = terminal_body("in_progress", nullptr, {});
    json::Value a = ev("response.created");
    a.set("response", body);
    push_event(std::move(a));
    json::Value b = ev("response.in_progress");
    b.set("response", std::move(body));
    push_event(std::move(b));
    return take();
}

void ResponsesStream::open_text(OpenItem &it, const char *type, long index) {
    it.id = id_ + "_" + type + "_" + std::to_string(index);
    it.type = type;
    it.index = index;
    json::Value item = json::Value::object();
    item.set("type", json::Value::string(type));
    item.set("id", json::Value::string(it.id));
    if (std::string(type) == "message") {
        item.set("role", json::Value::string("assistant"));
    } else {
        item.set("summary", json::Value::array());  // never invented
        item.set("encrypted_content", json::Value());
    }
    item.set("status", json::Value::string("in_progress"));
    item.set("content", json::Value::array());
    json::Value added = ev("response.output_item.added");
    added.set("output_index", json::Value::integer(index));
    added.set("item", item);
    push_event(std::move(added));
    json::Value part = json::Value::object();
    part.set("type", json::Value::string(std::string(type) == "message" ? "output_text" : "reasoning_text"));
    part.set("text", json::Value::string(""));
    if (std::string(type) == "message") part.set("annotations", json::Value::array());
    json::Value cp = ev("response.content_part.added");
    cp.set("item_id", json::Value::string(it.id));
    cp.set("output_index", json::Value::integer(index));
    cp.set("content_index", json::Value::integer(0));
    cp.set("part", std::move(part));
    push_event(std::move(cp));
    it.opened = true;
}

void ResponsesStream::close_text(OpenItem &it, const char *status) {
    if (!it.opened) return;
    const bool is_msg = it.type == "message";
    json::Value td = ev(is_msg ? "response.output_text.done" : "response.reasoning_text.done");
    td.set("item_id", json::Value::string(it.id));
    td.set("output_index", json::Value::integer(it.index));
    td.set("content_index", json::Value::integer(0));
    td.set("text", json::Value::string(it.text));
    push_event(std::move(td));
    json::Value pd = ev("response.content_part.done");
    pd.set("item_id", json::Value::string(it.id));
    pd.set("output_index", json::Value::integer(it.index));
    pd.set("content_index", json::Value::integer(0));
    json::Value part = json::Value::object();
    part.set("type", json::Value::string(is_msg ? "output_text" : "reasoning_text"));
    part.set("text", json::Value::string(it.text));
    if (is_msg) part.set("annotations", json::Value::array());
    pd.set("part", part);
    push_event(std::move(pd));
    json::Value item = json::Value::object();
    item.set("type", json::Value::string(it.type));
    item.set("id", json::Value::string(it.id));
    if (is_msg) item.set("role", json::Value::string("assistant"));
    else {
        item.set("summary", json::Value::array());
        item.set("encrypted_content", json::Value());
    }
    item.set("status", json::Value::string(status));
    json::Value parts = json::Value::array();
    parts.push(part);
    item.set("content", std::move(parts));
    json::Value id = ev("response.output_item.done");
    id.set("output_index", json::Value::integer(it.index));
    id.set("item", item);
    push_event(std::move(id));
    done_items_.push_back(std::move(item));
    it.opened = false;
    it.index = -1;
}

void ResponsesStream::on_reasoning(const std::string &text) {
    if (terminal_ || text.empty()) return;
    if (reasoning_.index < 0) {
        open_text(reasoning_, "reasoning", next_index_);
        ++next_index_;
    }
    reasoning_.text += text;
    json::Value d = ev("response.reasoning_text.delta");
    d.set("item_id", json::Value::string(reasoning_.id));
    d.set("output_index", json::Value::integer(reasoning_.index));
    d.set("content_index", json::Value::integer(0));
    d.set("delta", json::Value::string(text));
    push_event(std::move(d));
}

void ResponsesStream::on_content(const std::string &text) {
    if (terminal_ || text.empty()) return;
    close_text(reasoning_, "completed");  // the think block closed: the reasoning item is complete
    if (message_.index < 0) {
        open_text(message_, "message", next_index_);
        ++next_index_;
    }
    message_.text += text;
    json::Value d = ev("response.output_text.delta");
    d.set("item_id", json::Value::string(message_.id));
    d.set("output_index", json::Value::integer(message_.index));
    d.set("content_index", json::Value::integer(0));
    d.set("delta", json::Value::string(text));
    push_event(std::move(d));
}

void ResponsesStream::on_tool_call(const ToolCallOut &call) {
    if (terminal_) return;
    const json::Value *entry = nullptr;
    for (const auto &e : registry_.as_array("registry"))
        if (e.find("external")->as_string("external") == call.name) {
            entry = &e;
            break;
        }
    if (entry == nullptr)
        throw ApiError(500, "the model emitted a call to tool '" + call.name + "' that the request did not offer",
                       {}, "server_error");
    const json::Value *input = entry->find("kind")->as_string("kind") == "custom"
                                   ? call.arguments.find("input")
                                   : nullptr;
    if (entry->find("kind")->as_string("kind") == "custom" && (!input || !input->is_string()))
        // The parser accepts a name-only custom call (no required-param enforcement); the non-stream path
        // refuses it typed - the stream validates at the same seam (the review's BLOCK-1: flush_tools would
        // deref a missing input). A malformed call never reaches the wire; the route answers response.failed.
        throw ApiError(500, "the custom tool call '" + call.name + "' carries no string 'input'", {},
                       "server_error");
    // Buffered (the request's parser is the buffered one): the call is complete and registry-valid. Its item is
    // emitted at the terminal, with the terminal's status (a capped turn's calls are incomplete like its text).
    buffered_calls_.push_back(call);
}

void ResponsesStream::flush_tools(const char *item_status) {
    for (const ToolCallOut &call : buffered_calls_) {
        const json::Value *entry = nullptr;
        for (const auto &e : registry_.as_array("registry"))
            if (e.find("external")->as_string("external") == call.name) {
                entry = &e;
                break;
            }
        const bool is_custom = entry->find("kind")->as_string("kind") == "custom";
        const std::string &ns = entry->find("namespace")->as_string("namespace");
        const long index = next_index_++;
        const std::string item_id = id_ + (is_custom ? "_ctc_" : "_fc_") + std::to_string(index);
        const char *item_type = is_custom ? "custom_tool_call" : "function_call";
        const std::string plain_name = ns.empty() ? call.name : call.name.substr(ns.size() + 1);
        const std::string payload =
            is_custom ? call.arguments.find("input")->as_string("input") : call.arguments.dump();
        json::Value item = json::Value::object();
        item.set("type", json::Value::string(item_type));
        item.set("id", json::Value::string(item_id));
        item.set("status", json::Value::string("in_progress"));
        item.set("name", json::Value::string(plain_name));
        if (!ns.empty()) item.set("namespace", json::Value::string(ns));
        item.set("call_id", json::Value::string(call.id));
        item.set(is_custom ? "input" : "arguments", json::Value::string(""));
        json::Value added = ev("response.output_item.added");
        added.set("output_index", json::Value::integer(index));
        added.set("item", item);
        push_event(std::move(added));
        json::Value d =
            ev(is_custom ? "response.custom_tool_call_input.delta" : "response.function_call_arguments.delta");
        d.set("item_id", json::Value::string(item_id));
        d.set("output_index", json::Value::integer(index));
        d.set("call_id", json::Value::string(call.id));
        d.set("delta", json::Value::string(payload));
        push_event(std::move(d));
        json::Value dn =
            ev(is_custom ? "response.custom_tool_call_input.done" : "response.function_call_arguments.done");
        dn.set("item_id", json::Value::string(item_id));
        dn.set("output_index", json::Value::integer(index));
        dn.set("call_id", json::Value::string(call.id));
        dn.set(is_custom ? "input" : "arguments", json::Value::string(payload));
        push_event(std::move(dn));
        item.set("status", json::Value::string(item_status));
        item.set(is_custom ? "input" : "arguments", json::Value::string(payload));
        json::Value id = ev("response.output_item.done");
        id.set("output_index", json::Value::integer(index));
        id.set("item", item);
        push_event(std::move(id));
        done_items_.push_back(item);
    }
    buffered_calls_.clear();
}

json::Value ResponsesStream::terminal_body(const char *status, const GenerationResult *res,
                                           const std::string &failure) {
    json::Value body = json::Value::object();
    body.set("id", json::Value::string(id_));
    body.set("object", json::Value::string("response"));
    body.set("created_at", json::Value::integer(created_));
    body.set("model", json::Value::string(model_));
    body.set("status", json::Value::string(status));
    json::Value out = json::Value::array();
    for (const auto &it : done_items_) out.push(it);
    body.set("output", std::move(out));
    if (res) body.set("usage", usage_json_of(*res));
    if (!failure.empty()) {
        json::Value e = json::Value::object();
        e.set("message", json::Value::string(failure));
        e.set("type", json::Value::string("server_error"));
        body.set("error", std::move(e));
    }
    return body;
}

std::string ResponsesStream::finish(const char *status, const GenerationResult *res, const std::string &failure) {
    if (terminal_) return {};  // exactly one terminal event, always
    terminal_ = true;
    const char *item_status = std::string(status) == "completed" ? "completed" : "incomplete";
    close_text(reasoning_, item_status);
    close_text(message_, item_status);
    flush_tools(item_status);
    json::Value body = terminal_body(status, res, failure);
    if (std::string(status) == "incomplete") {
        json::Value d = json::Value::object();
        d.set("reason", json::Value::string("max_output_tokens"));
        body.set("incomplete_details", std::move(d));
    }
    json::Value e = ev(std::string(status) == "completed"     ? "response.completed"
                       : std::string(status) == "incomplete" ? "response.incomplete"
                                                             : "response.failed");
    e.set("response", std::move(body));
    push_event(std::move(e));
    return take();
}

std::string ResponsesStream::ev_completed(const GenerationResult &res) { return finish("completed", &res, {}); }
std::string ResponsesStream::ev_incomplete(const GenerationResult &res) { return finish("incomplete", &res, {}); }
std::string ResponsesStream::ev_failed(const std::string &message) { return finish("failed", nullptr, message); }

std::string ResponsesStream::take() {
    std::string out;
    out.swap(queue_);
    return out;
}

}  // namespace strix
