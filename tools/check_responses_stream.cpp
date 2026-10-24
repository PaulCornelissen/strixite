// The Responses SSE state machine, host only (STORY-08 / gate B8): event order and indices (sequence_number
// contiguous, output_index/item_id/content_index), the item lifecycle (added/part/deltas/done/item.done), tool
// items only when complete, SSE comments-free pure-data frames (the HTTP layer adds comments), terminal exactly
// once (completed / incomplete on the cap / failed on engine error + disconnect), no Chat chunks, no [DONE].
// The emitted streams are also dumped as jsonl for the pack's validate_event_stream (the contract the fixtures
// pin) - run by scripts/check_streams.py after this binary. No model, no GPU.
//
//   check_responses_stream [dump-dir]        (default /tmp/opencode)

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/output_parser.hpp"
#include "serve/responses.hpp"
#include "serve/responses_stream.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace strix;

namespace {

int failures = 0, checks = 0;
template <typename... A>
void ok(bool cond, A &&...a) {
    ++checks;
    if (!cond) {
        ++failures;
        std::ostringstream os;
        (os << ... << a);
        std::printf("FAIL: %s\n", os.str().c_str());
    }
}

const SamplingParams kDefaults;
const std::string kModel = "qwen3.8-flash-next";
std::string g_dump_dir = "/tmp/opencode";

ResponsesRequest request_with(const std::string &tools_json) {
    json::Value b = json::Value::parse("{\"model\":\"" + kModel + "\",\"input\":\"q\"" +
                                       (tools_json.empty() ? "" : ",\"tools\":" + tools_json) + "}");
    return parse_responses_request(b, kDefaults, kModel);
}

GenerationResult result(const std::string &finish, int64_t prompt = 100, int64_t cached = 10,
                        int64_t completion = 12, int64_t reasoning_toks = 4) {
    GenerationResult r;
    r.finish_reason = finish;
    r.prompt_tokens = prompt, r.cached_tokens = cached, r.completion_tokens = completion,
    r.reasoning_tokens = reasoning_toks;
    return r;
}

ToolCallOut call(const std::string &id, const std::string &name, const std::string &args_json) {
    ToolCallOut c;
    c.id = id, c.name = name, c.arguments = json::Value::parse(args_json);
    return c;
}

std::vector<json::Value> parse_frames(const std::string &bytes) {
    std::vector<json::Value> out;
    size_t pos = 0;
    while (pos < bytes.size()) {
        const size_t end = bytes.find("\n\n", pos);
        std::string frame = bytes.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = (end == std::string::npos) ? bytes.size() : end + 2;
        STRIX_CHECK(frame.size() > 6 && frame.compare(0, 6, "data: ") == 0, "non-data SSE frame: ", frame.substr(0, 40));
        out.push_back(json::Value::parse(frame.substr(6)));
    }
    return out;
}

std::string type_of(const json::Value &e) { return e.find("type")->as_string("type"); }

void check_common(const std::vector<json::Value> &events, const std::string &want_terminal) {
    long seq = -1;
    int terminals = 0;
    for (size_t k = 0; k < events.size(); ++k) {
        const json::Value &e = events[k];
        const std::string t = type_of(e);
        ok(e.find("sequence_number") && e.find("sequence_number")->as_int("s") == ++seq,
           t, ": contiguous sequence_number at ", k);
        ok(t.find("chat.completion") == std::string::npos, "no Chat event: ", t);
        if (t.rfind("response.", 0) == 0 && (t == "response.completed" || t == "response.incomplete" ||
                                             t == "response.failed"))
            ++terminals;
        ok(!(t == "[DONE]" || e.dump().find("[DONE]") != std::string::npos), "no Chat [DONE] sentinel");
        if (t == "response.output_item.added") {
            ok(e.find("item")->find("status")->as_string("status") == "in_progress", "added item is in_progress");
        }
        if (t == "response.output_item.done") {
            const json::Value &item = *e.find("item");
            ok(item.find("status")->as_string("status") != "in_progress", "done item is final");
        }
    }
    ok(terminals == 1, "exactly one terminal event, got ", terminals);
    ok(!events.empty() && type_of(events.back()) == want_terminal, "terminal is ", want_terminal, ", got ",
       events.empty() ? "none" : type_of(events.back()));
}

void dump(const std::string &name, const std::vector<json::Value> &events) {
    std::ofstream f(g_dump_dir + "/stream-" + name + ".jsonl", std::ios::binary);
    for (const auto &e : events) f << e.dump() << "\n";
    STRIX_CHECK(f.good(), "cannot dump stream-", name);
}

void check_mixed() {
    ResponsesRequest rr = request_with(
        "[{\"type\":\"function\",\"name\":\"exec_command\",\"parameters\":{\"type\":\"object\"}}]");
    ResponsesStream s("resp_m1", kModel, 1700000000, rr.tool_registry);
    std::string bytes = s.ev_created();
    s.on_reasoning("I should ");
    s.on_reasoning("look first.");
    s.on_content("Runni");
    s.on_content("ng now.");
    s.on_tool_call(call("call_9", "exec_command", "{\"cmd\":\"ls\"}"));
    bytes += s.take() + s.ev_completed(result("tool_calls"));
    auto events = parse_frames(bytes);
    check_common(events, "response.completed");
    // Order: created, in_progress, reasoning added/part/2 deltas/done/part.done/item.done, message same,
    // function added/delta/done/item.done, completed.
    const char *want[] = {"response.created",        "response.in_progress",
                          "response.output_item.added", "response.content_part.added",
                          "response.reasoning_text.delta", "response.reasoning_text.delta",
                          "response.reasoning_text.done", "response.content_part.done",
                          "response.output_item.done",
                          "response.output_item.added", "response.content_part.added",
                          "response.output_text.delta", "response.output_text.delta",
                          "response.output_text.done",  "response.content_part.done",
                          "response.output_item.done",
                          "response.output_item.added", "response.function_call_arguments.delta",
                          "response.function_call_arguments.done", "response.output_item.done",
                          "response.completed"};
    ok(events.size() == sizeof(want) / sizeof(want[0]), "event count, got ", events.size());
    bool order_ok = events.size() == sizeof(want) / sizeof(want[0]);
    for (size_t k = 0; order_ok && k < events.size(); ++k)
        order_ok = type_of(events[k]) == want[k];
    ok(order_ok, "full lifecycle order (reasoning -> message -> tool -> completed)");
    ok(type_of(events[2]).size() && events[2].find("output_index")->as_int("oi") == 0 &&
               events[9].find("output_index")->as_int("oi") == 1 &&
               events[16].find("output_index")->as_int("oi") == 2,
       "output_index 0,1,2 in order");
    ok(events[4].find("item_id")->as_string("id") == events[2].find("item")->find("id")->as_string("id"),
       "deltas carry their item_id");
    ok(events[4].find("content_index")->as_int("ci") == 0, "content_index 0");
    const json::Value &term = *events.back().find("response");
    ok(term.find("status")->as_string("status") == "completed", "terminal completed");
    ok(term.find("usage")->find("output_tokens_details")->find("reasoning_tokens")->as_int("rt") == 4,
       "terminal usage real");
    ok(term.find("output")->as_array("output").size() == 3, "terminal carries the completed items");
    ok(term.find("output")->as_array("output")[2].find("call_id")->as_string("cid") == "call_9",
       "tool item in the terminal body");
    dump("mixed", events);
}

void check_multi_tool_zero_text() {
    ResponsesRequest rr = request_with(
        "[{\"type\":\"function\",\"name\":\"a\",\"parameters\":{\"type\":\"object\"}},"
        "{\"type\":\"custom\",\"name\":\"apply_patch\"}]");
    ResponsesStream s("resp_m2", kModel, 1, rr.tool_registry);
    std::string bytes = s.ev_created();
    s.on_tool_call(call("call_1", "a", "{\"i\":1}"));
    s.on_tool_call(call("call_2", "apply_patch", "{\"input\":\"*** Begin Patch\\né\"}"));
    bytes += s.take() + s.ev_completed(result("tool_calls"));
    auto events = parse_frames(bytes);
    check_common(events, "response.completed");
    int f = 0, c = 0;
    for (const auto &e : events) {
        if (type_of(e) == "response.output_item.added") {
            const std::string &it = e.find("item")->find("type")->as_string("t");
            if (it == "function_call") ++f;
            if (it == "custom_tool_call") ++c;
        }
    }
    ok(f == 1 && c == 1, "function_call + custom_tool_call items");
    const json::Value &out = *events.back().find("response")->find("output");
    ok(out.as_array("o").size() == 2, "zero text: two tool items only");
    ok(out.as_array("o")[1].find("input")->as_string("in") == "*** Begin Patch\né",
       "custom input byte-for-byte on the wire");
    ok(out.as_array("o")[0].find("arguments")->as_string("ar") == "{\"i\":1}", "function arguments JSON text");
    dump("multi_tool", events);
}

void check_incomplete_and_failed() {
    ResponsesRequest rr = request_with("");
    ResponsesStream s("resp_i1", kModel, 1, rr.tool_registry);
    std::string bytes = s.ev_created();
    s.on_content("truncat");
    s.on_content("ing…");
    bytes += s.take() + s.ev_incomplete(result("length"));
    auto events = parse_frames(bytes);
    check_common(events, "response.incomplete");
    const json::Value &body = *events.back().find("response");
    ok(body.find("status")->as_string("status") == "incomplete", "status incomplete");
    ok(body.find("incomplete_details")->find("reason")->as_string("r") == "max_output_tokens",
       "incomplete_details.reason");
    ok(body.find("output")->as_array("o").back().find("status")->as_string("st") == "incomplete",
       "the last item is marked incomplete (contract)");
    dump("incomplete", events);

    ResponsesStream f("resp_f1", kModel, 1, rr.tool_registry);
    bytes = f.ev_created();
    f.on_content("gone");
    bytes += f.take() + f.ev_completed(result("stop"));  // second terminal: ignored
    bytes += f.take() + f.ev_failed("late failure");
    auto fevents = parse_frames(bytes);
    check_common(fevents, "response.completed");  // exactly one terminal: the late failure emitted nothing

    ResponsesStream g("resp_g1", kModel, 1, rr.tool_registry);
    bytes = g.ev_created();
    g.on_content("partial");
    bytes += g.take() + g.ev_failed("engine exploded");
    auto gevents = parse_frames(bytes);
    check_common(gevents, "response.failed");
    ok(gevents.back().find("response")->find("error")->find("message")->as_string("m") == "engine exploded",
       "failed carries the error");
    ok(gevents.back().find("response")->find("output")->as_array("o").back().find("status")
               ->as_string("st") == "incomplete",
       "items closed by a failure carry the contract's incomplete status");
    dump("failed_engine", gevents);
}

void check_unknown_tool_and_guards() {
    ResponsesRequest rr = request_with("[{\"type\":\"function\",\"name\":\"a\",\"parameters\":{\"type\":\"object\"}}]");
    ResponsesStream s("resp_u1", kModel, 1, rr.tool_registry);
    s.ev_created();
    bool threw = false;
    try {
        s.on_tool_call(call("call_x", "not_offered", "{}"));
    } catch (const ApiError &e) {
        threw = e.status == 500 && e.type == "server_error";
    }
    ok(threw, "a call for an un-offered tool throws (never published)");
    const std::string term = s.ev_failed("the model emitted a call to tool 'not_offered' that the request did not offer");
    auto events = parse_frames(term);
    ok(events.size() == 1 && type_of(events[0]) == "response.failed", "typed terminal after the refusal");
    ok(s.ev_completed(result("stop")).empty(), "terminal is exactly once");
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1) g_dump_dir = argv[1];
    try {
        check_mixed();
        check_multi_tool_zero_text();
        check_incomplete_and_failed();
        check_unknown_tool_and_guards();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
