// The non-streaming /v1/responses serializer, host only (STORY-07 / gate B7): the response JSON for every
// generation outcome - text + raw reasoning, tool-only, mixed, two tools, output-cap (incomplete, never a fake
// success), engine failure (typed 500, never a body Codex acts on), unknown-call rejection (nothing executable
// is published), custom calls byte-for-byte, namespaced restore, ids (item ids distinct from call ids) and
// usage taken straight from GenerationResult (input/cached/output/reasoning).
//
//   check_responses_body

#include "common/check.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/output_parser.hpp"
#include "serve/responses.hpp"
#include "serve/responses_body.hpp"

#include <cstdio>
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

// A request whose registry defines exactly the tools a case needs.
ResponsesRequest request_with(const std::string &tools_json) {
    json::Value b = json::Value::parse("{\"model\":\"" + kModel + "\",\"input\":\"q\"" +
                                       (tools_json.empty() ? "" : ",\"tools\":" + tools_json) + "}");
    return parse_responses_request(b, kDefaults, kModel);
}

GenerationResult result(const std::string &finish, int64_t prompt = 100, int64_t cached = 40,
                        int64_t completion = 12, int64_t reasoning_toks = 5, const std::string &error = {}) {
    GenerationResult r;
    r.finish_reason = finish;
    r.error = error;
    r.prompt_tokens = prompt, r.cached_tokens = cached, r.completion_tokens = completion,
    r.reasoning_tokens = reasoning_toks;
    return r;
}

ToolCallOut call(const std::string &id, const std::string &name, const std::string &args_json) {
    ToolCallOut c;
    c.id = id, c.name = name, c.arguments = json::Value::parse(args_json);
    return c;
}

const json::Value *item(const json::Value &body, size_t k) {
    const json::Value *out = body.find("output");
    if (!out || k >= out->as_array("output").size()) return nullptr;
    return &out->as_array("output")[k];
}

// --- the six acceptance shapes ------------------------------------------------------------------------------------

void check_text_with_reasoning() {
    ResponsesRequest rr = request_with("");
    json::Value b = responses_body(rr, "resp_a1", 1700000000, "  thinking hard  ", "The answer.", {},
                                   result("stop"));
    ok(b.find("object")->as_string("object") == "response", "object=response");
    ok(b.find("id")->as_string("id") == "resp_a1", "response id");
    ok(b.find("created_at")->as_int("created_at") == 1700000000, "created_at");
    ok(b.find("model")->as_string("model") == kModel, "model is the served id");
    ok(b.find("status")->as_string("status") == "completed", "status completed");
    ok(b.find("store") && b.find("store")->as_bool("store") == false, "store:false (stateless)");
    const json::Value *rs = item(b, 0);
    ok(rs && rs->find("type")->as_string("type") == "reasoning", "reasoning item first");
    if (rs) {
        ok(rs->find("summary") && rs->find("summary")->as_array("summary").empty(), "summary is the empty array (never invented)");
        ok(rs->find("encrypted_content") && rs->find("encrypted_content")->is_null(),
           "encrypted_content is null (no key, no vault - disclosed)");
        const json::Value &c = *rs->find("content");
        ok(c.as_array("c").size() == 1 && c.as_array("c")[0].find("type")->as_string("type") == "reasoning_text" &&
                   c.as_array("c")[0].find("text")->as_string("text") == "  thinking hard  ",
           "raw reasoning rides as reasoning_text, unmodified");
    }
    const json::Value *msg = item(b, 1);
    ok(msg && msg->find("type")->as_string("type") == "message" &&
               msg->find("role")->as_string("role") == "assistant" &&
               msg->find("status")->as_string("status") == "completed",
       "message item, assistant, completed");
    ok(msg && msg->find("content")->as_array("content")[0].find("type")->as_string("type") == "output_text" &&
               msg->find("content")->as_array("content")[0].find("text")->as_string("text") == "The answer.",
       "output_text content part");
    ok(b.find("output_text")->as_string("output_text") == "The answer.", "output_text convenience field");
    ok(item(b, 2) == nullptr, "exactly the items produced");
    // Usage straight from GenerationResult.
    const json::Value &u = *b.find("usage");
    ok(u.find("input_tokens")->as_int("input_tokens") == 100, "input_tokens = prompt_tokens");
    ok(u.find("input_tokens_details")->find("cached_tokens")->as_int("cached") == 40, "cached_tokens");
    ok(u.find("output_tokens")->as_int("output_tokens") == 12, "output_tokens = completion_tokens");
    ok(u.find("output_tokens_details")->find("reasoning_tokens")->as_int("reasoning") == 5, "reasoning_tokens");
    ok(u.find("total_tokens")->as_int("total") == 112, "total = in + out");
}

void check_tool_only() {
    ResponsesRequest rr = request_with("[{\"type\":\"function\",\"name\":\"exec_command\",\"parameters\":{\"type\":\"object\",\"properties\":{\"cmd\":{\"type\":\"string\"}}}}]");
    json::Value b = responses_body(rr, "resp_b1", 1, "", "",
                                   {call("call_x1", "exec_command", "{\"cmd\":\"ls\"}")}, result("tool_calls"));
    ok(b.find("status")->as_string("status") == "completed", "tool_calls finish = completed");
    const json::Value *fc = item(b, 0);
    ok(fc && fc->find("type")->as_string("type") == "function_call", "function_call item, no message items for silence");
    if (fc) {
        ok(fc->find("call_id")->as_string("call_id") == "call_x1", "the parser's call_id preserved");
        ok(fc->find("name")->as_string("name") == "exec_command", "name");
        ok(fc->find("status")->as_string("status") == "completed", "item completed");
        ok(fc->find("arguments")->as_string("arguments") == "{\"cmd\":\"ls\"}", "arguments as JSON text");
        ok(fc->find("id")->as_string("id") != fc->find("call_id")->as_string("call_id"),
           "item id is not the call id (separate namespaces)");
        ok(fc->find("id")->as_string("id").find("call_") == std::string::npos, "call id not reused as item id");
    }
}

void check_mixed() {
    ResponsesRequest rr = request_with("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]");
    json::Value b = responses_body(rr, "resp_c1", 1, "think", "Working on it.",
                                   {call("call_1", "f", "{}")}, result("tool_calls"));
    ok(item(b, 0)->find("type")->as_string("type") == "reasoning" &&
               item(b, 1)->find("type")->as_string("type") == "message" &&
               item(b, 2)->find("type")->as_string("type") == "function_call",
       "reasoning, message, function_call in order");
}

void check_two_tools() {
    ResponsesRequest rr = request_with(
        "[{\"type\":\"function\",\"name\":\"a\",\"parameters\":{\"type\":\"object\"}},"
        "{\"type\":\"function\",\"name\":\"b\",\"parameters\":{\"type\":\"object\"}}]");
    json::Value b = responses_body(rr, "resp_d1", 1, "", "",
                                   {call("call_p", "a", "{\"i\":1}"), call("call_q", "b", "{\"s\":\"x\"}")},
                                   result("tool_calls"));
    ok(item(b, 0) && item(b, 1) && !item(b, 2), "two call items");
    ok(item(b, 0)->find("call_id")->as_string("call_id") == "call_p" &&
               item(b, 1)->find("call_id")->as_string("call_id") == "call_q",
       "each keeps its own call_id");
    ok(item(b, 0)->find("id")->as_string("id") != item(b, 1)->find("id")->as_string("id"), "distinct item ids");
    ok(item(b, 1)->find("arguments")->as_string("arguments") == "{\"s\":\"x\"}", "second call arguments");
}

void check_incomplete_on_output_cap() {
    ResponsesRequest rr = request_with("");
    GenerationResult res = result("length", 100, 0, 64, 60);
    json::Value b = responses_body(rr, "resp_e1", 1, "thoughts", "cut off mid sen", {}, res);
    ok(b.find("status")->as_string("status") == "incomplete", "token cap = incomplete, not completed");
    ok(b.find("incomplete_details") &&
               b.find("incomplete_details")->find("reason")->as_string("reason") == "max_output_tokens",
       "incomplete_details.reason");
    ok(item(b, 1)->find("content")->as_array("content")[0].find("text")->as_string("text") == "cut off mid sen",
       "the partial text is what it is: present, under incomplete status");
    // A completed call plus the cap: the call is real (it closed), the status is still incomplete.
    ResponsesRequest rr2 = request_with("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]");
    json::Value b2 = responses_body(rr2, "resp_e2", 1, "", "", {call("call_1", "f", "{}")}, result("length"));
    ok(b2.find("status")->as_string("status") == "incomplete", "capped turn with a completed call: incomplete");
    ok(item(b2, 0)->find("type")->as_string("type") == "function_call", "the completed call is published");
}

void expect_500(const ResponsesRequest &rr, const std::string &reasoning, const std::string &content,
                const std::vector<ToolCallOut> &calls, const GenerationResult &res, const std::string &what) {
    try {
        responses_body(rr, "resp_x", 1, reasoning, content, calls, res);
        ok(false, what, ": built a body, expected a typed 500");
    } catch (const ApiError &e) {
        ok(e.status == 500 && e.type == "server_error", what, ": got ", e.status, " '", e.what(), "'");
    }
}

void check_failures() {
    ResponsesRequest rr = request_with("");
    expect_500(rr, "", "half an answer", {}, result("error", 10, 0, 1, 0, "engine exploded"),
               "engine error: typed failure, no success body");
    expect_500(rr, "", "gone", {}, result("cancelled"), "cancelled: typed failure");
    expect_500(rr, "", "", {}, result("stop"), "an empty turn is not a success to report");
    // A call for a tool the request never offered: publishing it could be acted on blindly - refuse.
    expect_500(rr, "", "", {call("call_z", "not_offered", "{}")}, result("tool_calls"),
               "unknown tool call name: refused (registry is the allowlist)");
    // A custom call whose internal input argument is missing.
    ResponsesRequest rr2 = request_with("[{\"type\":\"custom\",\"name\":\"apply_patch\"}]");
    expect_500(rr2, "", "", {call("call_1", "apply_patch", "{}")}, result("tool_calls"),
               "custom call without input: refused");
}

void check_custom_and_namespaced() {
    ResponsesRequest rr = request_with(
        "[{\"type\":\"custom\",\"name\":\"apply_patch\"},"
        "{\"type\":\"namespace\",\"name\":\"multi_agent_v1\",\"tools\":[{\"type\":\"function\",\"name\":\"spawn_agent\",\"parameters\":{\"type\":\"object\"}}]}]");
    json::Value b = responses_body(rr, "resp_f1", 1, "", "",
                                   {call("call_1", "apply_patch", "{\"input\":\"*** Begin Patch\\né\\n\"}"),
                                    call("call_2", "multi_agent_v1/spawn_agent", "{}")},
                                   result("tool_calls"));
    const json::Value *ctc = item(b, 0);
    ok(ctc && ctc->find("type")->as_string("type") == "custom_tool_call", "custom_tool_call item");
    ok(ctc && ctc->find("input")->as_string("input") == "*** Begin Patch\né\n", "input byte-for-byte");
    ok(ctc && ctc->find("call_id")->as_string("call_id") == "call_1", "custom keeps the call id");
    const json::Value *fc = item(b, 1);
    ok(fc && fc->find("type")->as_string("type") == "function_call" &&
               fc->find("name")->as_string("name") == "spawn_agent" &&
               fc->find("namespace")->as_string("namespace") == "multi_agent_v1",
       "namespaced call restores namespace + plain name");
}

}  // namespace

int main() {
    try {
        check_text_with_reasoning();
        check_tool_only();
        check_mixed();
        check_two_tools();
        check_incomplete_on_output_cap();
        check_failures();
        check_custom_and_namespaced();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
