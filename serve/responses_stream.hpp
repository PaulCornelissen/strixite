#pragma once

// The typed Responses SSE state machine (the stream wire Codex reads): consumes the buffered parser's events
// and emits ordered SSE frames - response.created/in_progress, per-item output_item.added / content_part.added /
// (reasoning|output)_text.delta / .done / content_part.done / output_item.done, tool items only once COMPLETE
// (the request runs the buffered parser: a partial call never reaches the wire), and exactly one terminal -
// response.completed, response.incomplete (output cap, last item incomplete) or response.failed. Consecutive
// sequence_number, output_index, item_id, content_index throughout; no Chat chunks, no [DONE] sentinel.
// The frame contract is the pack's stream fixtures + validate_event_stream (scripts/), which the host tests
// run against the emitted streams.

#include "serve/engine.hpp"         // GenerationResult
#include "serve/openai.hpp"         // ApiError
#include "serve/output_parser.hpp"  // ToolCallOut
#include "serve/json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

class ResponsesStream {
public:
    ResponsesStream(std::string resp_id, std::string model, int64_t created, json::Value registry,
                    bool parallel_tool_calls = true);

    std::string ev_created();  // response.created + response.in_progress frames (starts the stream)
    void on_reasoning(const std::string &text);
    void on_content(const std::string &text);
    void on_tool_call(const ToolCallOut &call);  // registry allowlist: an unknown name throws ApiError 500
                                                 // (validated now; the item is emitted at the terminal, buffered)
    // Terminal events (each idempotent-guarded: the state machine emits its single terminal, period).
    std::string ev_completed(const GenerationResult &res);
    std::string ev_incomplete(const GenerationResult &res);  // the output cap
    std::string ev_failed(const std::string &message);       // engine failure / cancellation

    std::string take();  // queued "data: ...\n\n" frames (SSE comments are the HTTP layer's)
    bool terminal() const { return terminal_; }

private:
    struct OpenItem {  // the live text item (reasoning XOR message, at most one open at a time)
        std::string id, type;
        long index = -1;
        bool opened = false;  // added/part frames sent
        std::string text;
    };
    void open_text(OpenItem &it, const char *type, long index);
    void close_text(OpenItem &it, const char *status);  // .done part/item frames; records the item
    void flush_tools(const char *item_status);          // the buffered calls, with the terminal's status
    json::Value terminal_body(const char *status, const GenerationResult *res, const std::string &failure);
    void push_event(json::Value v);  // stamps sequence_number and queues the frame
    std::string finish(const char *status, const GenerationResult *res, const std::string &failure);

    std::string id_, model_;
    int64_t created_;
    json::Value registry_;
    long seq_ = 0;
    long next_index_ = 0;  // next output_index
    OpenItem reasoning_, message_;
    std::vector<ToolCallOut> buffered_calls_;  // validated at on_tool_call, emitted at the terminal
    std::vector<json::Value> done_items_;      // the output array (terminal body must equal it)
    std::string queue_;
    bool terminal_ = false, started_ = false;
    bool parallel_tool_calls_ = true;
};

}  // namespace strix
