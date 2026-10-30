// The /v1/responses multi-turn history replay, host only (STORY-06 / gate B6): function_call / custom_tool_call
// items and their *_output results rebuild the ordered assistant(tool_calls) + tool Chat messages with the call
// ids, names, arguments and full output text intact; parallel and out-of-order outputs pair by call_id, never by
// position; duplicate/dangling/early outputs are refused (nothing pairs to a wrong call); a historical call whose
// tool is not currently offered replays; reasoning items are display metadata (never invented, never faked); the
// normalization is deterministic to the rendered prompt. No model, no GPU, no tokenizer.
//
//   check_responses_history [codex-history-2turn.json]

#include "common/check.hpp"
#include "serve/chat_template.hpp"
#include "serve/json.hpp"
#include "serve/openai.hpp"
#include "serve/responses.hpp"

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
const std::string kHead = "{\"model\":\"" + kModel + "\",\"input\":[";
const std::string kTail = "]}";

ResponsesRequest parse(const std::string &json) { return parse_responses_request(json::Value::parse(json), kDefaults, kModel); }

const json::Value &at(const ResponsesRequest &r, size_t k) { return r.chat.messages.as_array("m")[k]; }
std::string role(const ResponsesRequest &r, size_t k) { return at(r, k).find("role")->as_string("role"); }
std::string content(const ResponsesRequest &r, size_t k) { return at(r, k).find("content")->as_string("content"); }
std::string head(const std::string &items) { return kHead + items + kTail; }

const json::Value *call0(const ResponsesRequest &r, size_t msg) {
    const json::Value *tc = at(r, msg).find("tool_calls");
    if (!tc || !tc->is_array() || tc->as_array("tc").empty()) return nullptr;
    return &tc->as_array("tc")[0];
}

// --- the single-call round ------------------------------------------------------------------------------------

void check_single_call() {
    ResponsesRequest r = parse(head(
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"ls\"},"
        "{\"type\":\"function_call\",\"name\":\"exec_command\",\"call_id\":\"c1\",\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"a.txt\\nb.txt\\n\"},"
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"thanks\"}"));
    ok(r.chat.messages.as_array("m").size() == 4, "user, assistant(call), tool, user, got ",
       r.chat.messages.as_array("m").size());
    ok(role(r, 1) == "assistant" && role(r, 2) == "tool" && role(r, 3) == "user", "roles in order");
    const json::Value *c = call0(r, 1);
    ok(c != nullptr, "assistant carries the tool call");
    if (c) {
        ok(c->find("id")->as_string("id") == "c1", "call_id preserved");
        ok(c->find("function")->find("name")->as_string("name") == "exec_command", "name preserved");
        ok(c->find("function")->find("arguments")->find("cmd")->as_string("cmd") == "ls", "arguments parsed object");
    }
    ok(at(r, 2).find("tool_call_id")->as_string("tool_call_id") == "c1", "output names its call");
    ok(content(r, 2) == "a.txt\nb.txt\n", "full output text, newlines intact");
    ok(content(r, 3) == "thanks", "the follow-up user message survives");
}

// --- parallel calls, delayed (out-of-order) outputs ------------------------------------------------------------

void check_parallel_and_delayed() {
    ResponsesRequest r = parse(head(
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"two\"},"
        "{\"type\":\"function_call\",\"name\":\"f\",\"call_id\":\"c1\",\"arguments\":\"{}\"},"
        "{\"type\":\"function_call\",\"name\":\"g\",\"call_id\":\"c2\",\"arguments\":\"{}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c2\",\"output\":\"second\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"first\"}"));
    ok(role(r, 1) == "assistant", "one assistant turn");
    const json::Value *tc = at(r, 1).find("tool_calls");
    ok(tc && tc->as_array("tc").size() == 2, "two parallel calls in the one assistant message, got ",
       tc ? (long)tc->as_array("tc").size() : -1);
    if (tc && tc->as_array("tc").size() == 2) {
        ok(tc->as_array("tc")[0].find("id")->as_string("id") == "c1" &&
                   tc->as_array("tc")[1].find("id")->as_string("id") == "c2",
           "call order is the request order");
    }
    ok(role(r, 2) == "tool" && content(r, 2) == "second" &&
               at(r, 2).find("tool_call_id")->as_string("tool_call_id") == "c2",
       "the delayed output pairs by call_id (c2 answered first)");
    ok(role(r, 3) == "tool" && content(r, 3) == "first", "c1's output follows, paired by id");
}

// --- custom freeform round: raw input + text output --------------------------------------------------------------

void check_custom_round() {
    ResponsesRequest r = parse(head(
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"patch\"},"
        "{\"type\":\"custom_tool_call\",\"name\":\"apply_patch\",\"call_id\":\"c9\",\"input\":\"*** Begin Patch\\n-é\\n+x\\t\\n*** End Patch\"},"
        "{\"type\":\"custom_tool_call_output\",\"call_id\":\"c9\",\"output\":\"Success.\"}"));
    const json::Value *c = call0(r, 1);
    ok(c != nullptr, "custom call on the assistant message");
    if (c) {
        ok(c->find("function")->find("name")->as_string("name") == "apply_patch", "name kept");
        const json::Value *in = c->find("function")->find("arguments")->find("input");
        ok(in && in->as_string("input") == "*** Begin Patch\n-é\n+x\t\n*** End Patch",
           "custom_tool_call.input byte-for-byte in the replay");
    }
    ok(content(r, 2) == "Success.", "the custom result text is the tool message");
}

// --- pairing integrity: every refusal is explicit, nothing silently pairs ----------------------------------------

void expect_400(const std::string &json, const std::string &what) {
    try {
        parse(json);
        ok(false, what, ": accepted, expected a 400 naming input");
    } catch (const ApiError &e) {
        ok(e.status == 400 && e.param == "input", what, ": got ", e.status, " param '", e.param, "'");
    }
}

void check_pairing_rules() {
    const std::string user = "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},";
    const std::string call1 = "{\"type\":\"function_call\",\"name\":\"f\",\"call_id\":\"c1\",\"arguments\":\"{}\"},";
    const std::string out1 = "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"x\"}";
    expect_400(head(user + call1 +
                         "{\"type\":\"function_call\",\"name\":\"g\",\"call_id\":\"c1\",\"arguments\":\"{}\"}," + out1),
               "duplicate call_id refused");
    expect_400(head(user + call1 + out1 + "," + out1), "a call answered twice refused");
    expect_400(head(user + call1 + "{\"type\":\"message\",\"role\":\"user\",\"content\":\"next\"}"),
               "a call with no output before the next message refused (dangling)");
    expect_400(head(user + out1), "an output with no call refused");
    expect_400(head(user + "{\"type\":\"function_call\",\"name\":\"f\",\"call_id\":\"c1\",\"arguments\":\"not json\"}," +
                        out1),
               "unparseable arguments refused at parse time (not at render)");
    expect_400(head(user + "{\"type\":\"custom_tool_call\",\"name\":\"p\",\"input\":5}," + out1),
               "a custom call with a non-string input refused");
    expect_400(head(user + "{\"type\":\"function_call\",\"name\":\"f\",\"arguments\":\"{}\"}," + out1),
               "a call without call_id refused");
    // Trailing calls (the last turn ended on calls) are unambiguous and replay fine:
    ResponsesRequest tail = parse(head(user + call1.substr(0, call1.size() - 1)));  // the last item has no comma
    ok(role(tail, 1) == "assistant" && call0(tail, 1) != nullptr, "a trailing unanswered call is kept, not dropped");
    // The shape check needs a live registry - and only then:
    expect_400("{\"model\":\"" + kModel + "\",\"input\":[" + user +
                   "{\"type\":\"function_call\",\"name\":\"p\",\"call_id\":\"c1\",\"arguments\":\"{}\"}," + out1 +
                   "],\"tools\":[{\"type\":\"custom\",\"name\":\"p\"}]}",
               "a function_call for a currently-offered custom tool refused (kind mismatch)");
}

// --- the history tool may be gone from the current request ---------------------------------------------------------

void check_historical_tool_absent() {
    ResponsesRequest r = parse("{\"model\":\"" + kModel + "\",\"input\":[" +
                              "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},"
                              "{\"type\":\"function_call\",\"name\":\"old_tool\",\"call_id\":\"c1\",\"arguments\":\"{}\"},"
                              "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"was fine\"}],"
                              "\"tools\":[]}");
    ok(r.chat.tools.is_null(), "no tools offered now");
    ok(r.tool_registry.as_array("reg").empty(), "empty registry");
    ok(call0(r, 1) && call0(r, 1)->find("function")->find("name")->as_string("name") == "old_tool",
       "the historical call replays without a live tool");
    ok(content(r, 2) == "was fine", "its output replays");
}

// --- reasoning items: metadata, ignored; state untouched ----------------------------------------------------------------

void check_reasoning_metadata() {
    ResponsesRequest r = parse(head(
        "{\"type\":\"message\",\"role\":\"user\",\"content\":\"q\"},"
        "{\"type\":\"reasoning\",\"id\":\"r1\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"short\"}],\"encrypted_content\":\"BLOB\"},"
        "{\"type\":\"function_call\",\"name\":\"f\",\"call_id\":\"c1\",\"arguments\":\"{}\"},"
        "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"x\"},"
        "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\"}]}"));
    ok(r.chat.messages.as_array("m").size() == 4, "reasoning item adds no message (never invented reasoning), got ",
       r.chat.messages.as_array("m").size());
    ok(role(r, 1) == "assistant" && content(r, 3) == "done", "pairing survives across the reasoning item");
}

// --- the simulated two-round Codex transcript, end to end ----------------------------------------------------------------

void check_two_turn_fixture(const json::Value &fixture) {
    ResponsesRequest r = parse_responses_request(fixture, kDefaults, kModel);
    const auto &msgs = r.chat.messages.as_array("m");
    // [system(developer), user, assistant(2 calls), tool(c2), tool(c1), assistant(custom call), tool(c3), assistant text, user]
    ok(msgs.size() == 9, "9 chat messages, got ", msgs.size());
    if (msgs.size() != 9) return;
    ok(role(r, 0) == "system" && role(r, 1) == "user", "the opening turns");
    ok(role(r, 2) == "assistant" && at(r, 2).find("tool_calls")->as_array("tc").size() == 2,
       "both exec_command calls on one assistant turn");
    ok(at(r, 3).find("tool_call_id")->as_string("tool_call_id") == "call_2" && content(r, 3) == "ok\n",
       "call_2 answered first (delayed order kept)");
    ok(at(r, 4).find("tool_call_id")->as_string("tool_call_id") == "call_1" && content(r, 4) == "",
       "call_1's empty output preserved");
    ok(role(r, 5) == "assistant" && call0(r, 5)->find("function")->find("name")->as_string("name") == "apply_patch",
       "the custom call is its own assistant turn");
    ok(call0(r, 5)->find("function")->find("arguments")->find("input")->as_string("input").find("patched é —") !=
               std::string::npos,
       "the patch text (with Unicode) rides byte-for-byte");
    ok(content(r, 6) == "Success. Updated the following files:\nM notes.md", "the custom result replays");
    ok(content(r, 7) == "Created marker.txt (verified: ok) and patched notes.md.", "the final answer text");
    ok(content(r, 8) == "Now remove the marker file.", "the follow-up user turn");

    // Determinism to the rendered prompt: the same transcript renders the same string twice.
    const json::Value *tools = r.chat.tools.is_null() ? nullptr : &r.chat.tools;
    ChatTemplateOptions opt;
    opt.enable_thinking = false;  // the prompt shape is the point, not the thinking block
    const std::string p1 = render_chat(r.chat.messages, tools, opt);
    ResponsesRequest r2 = parse_responses_request(fixture, kDefaults, kModel);
    const std::string p2 = render_chat(r2.chat.messages, r2.chat.tools.is_null() ? nullptr : &r2.chat.tools, opt);
    ok(p1 == p2, "the same transcript renders the identical prompt (prompt-cache friendly)");
    ok(p1.find("<tool_call>\n<function=exec_command>") != std::string::npos, "the calls render in the model's tool XML");
    ok(p1.find("<tool_response>") != std::string::npos && p1.find("ok\n") != std::string::npos,
       "the outputs render as tool responses");
}


// --- external review B2 (2026-10-27): namespaced call items keep their identity in history --------------------
// A response emits a namespace child as {namespace:"ns", name:"plain"}; the flat model-facing name is
// "ns/plain" (the registry key, what the model wrote in <function=ns/plain>). Replaying the emitted item
// must restore the FLAT name in the ChatRequest and in the rendered prompt - the plain name alone changes
// the tool's identity (and can collide with a top-level tool of that plain name).

void check_namespaced_history() {
    // The tool as Codex registers it (the capture's multi_agent_v1/spawn_agent) and the item the route emitted.
    const std::string tools =
        R"JX("tools":[{"type":"namespace","name":"multi_agent_v1","tools":[{"type":"function","name":"spawn_agent",)JX"
        R"JX("parameters":{"type":"object","properties":{}}}]}])JX";
    const std::string ns_call =
        R"JX({"type":"function_call","namespace":"multi_agent_v1","name":"spawn_agent","call_id":"c1","arguments":"{}"},)JX";
    const std::string out1 = R"JX({"type":"function_call_output","call_id":"c1","output":"agent 7 started"},)JX";
    const std::string user = R"JX({"type":"message","role":"user","content":"spawn a helper"},)JX";

    {  // the review's regression: replay the emitted {namespace, name} item -> flat name in the ChatRequest
        const std::string q = R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user + ns_call + out1 +
                              R"JX({"type":"message","role":"user","content":"and report"}],)JX" + tools + "}";
        ResponsesRequest r = parse(q);
        const json::Value *c = call0(r, 1);
        ok(c != nullptr, "the replayed namespaced call is on the assistant turn");
        if (c)
            ok(c->find("function")->find("name")->as_string("name") == "multi_agent_v1/spawn_agent",
               "the namespaced replay restores the flat model-facing name, got ",
               c->find("function")->find("name")->as_string("name"));
        // and the model actually sees it: the rendered prompt carries the flat name, not the plain one
        ChatTemplateOptions opt;
        opt.enable_thinking = false;
        const std::string p = render_chat(r.chat.messages, &r.chat.tools, opt);
        ok(p.find("<function=multi_agent_v1/spawn_agent>") != std::string::npos,
           "the rendered prompt carries the flat namespaced name");
        ok(p.find("<function=spawn_agent>\n") == std::string::npos,
           "the plain name alone never renders for a namespaced tool");
        ok(p.find("agent 7 started") != std::string::npos, "the output replays on the same round");
    }
    {  // the tool is NOT currently offered: history identity survives all the same (the review's explicit ask)
        const std::string q = R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user + ns_call + out1.substr(0, out1.size() - 1) + "]}";
        ResponsesRequest r = parse(q);
        const json::Value *c = call0(r, 1);
        ok(c && c->find("function")->find("name")->as_string("name") == "multi_agent_v1/spawn_agent",
           "a historical namespaced call keeps its flat name with no registry to consult");
    }
    {  // collision handling: a client that replays the FLAT name with namespace set must not double the prefix
        const std::string q = R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                              R"JX({"type":"function_call","namespace":"multi_agent_v1","name":"multi_agent_v1/spawn_agent",)JX"
                              R"JX("call_id":"c1","arguments":"{}"},)JX" + out1.substr(0, out1.size() - 1) + "]," + tools + "}";
        ResponsesRequest r = parse(q);
        const json::Value *c = call0(r, 1);
        ok(c && c->find("function")->find("name")->as_string("name") == "multi_agent_v1/spawn_agent",
           "flat name + its own namespace is idempotent, got ",
           (c ? c->find("function")->find("name")->as_string("name") : "?"));
    }
    {  // a flat replay WITHOUT the namespace field is the same tool (the registry key form): accepted as-is
        const std::string q = R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                              R"JX({"type":"function_call","name":"multi_agent_v1/spawn_agent","call_id":"c1","arguments":"{}"},)JX"
                              + out1.substr(0, out1.size() - 1) + "]," + tools + "}";
        ResponsesRequest r = parse(q);
        const json::Value *c = call0(r, 1);
        ok(c && c->find("function")->find("name")->as_string("name") == "multi_agent_v1/spawn_agent",
           "a flat-form replay matches the namespaced registry entry");
    }
    {  // namespace:"" is the absent form (SDKs send explicit empties/nulls)
        const std::string q = R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                              R"JX({"type":"function_call","namespace":"","name":"plain","call_id":"c1","arguments":"{}"},)JX"
                              + out1.substr(0, out1.size() - 1) + "]}";
        ResponsesRequest r = parse(q);
        const json::Value *c = call0(r, 1);
        ok(c && c->find("function")->find("name")->as_string("name") == "plain", "an empty namespace is a no-op");
        const std::string qn = R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                               R"JX({"type":"function_call","namespace":null,"name":"plain","call_id":"c1","arguments":"{}"},)JX"
                               + out1.substr(0, out1.size() - 1) + "]}";
        ResponsesRequest rn = parse(qn);
        const json::Value *cn = call0(rn, 1);
        ok(cn && cn->find("function")->find("name")->as_string("name") == "plain", "a null namespace is a no-op");
    }
    {  // the registry shape check runs on the FLAT name: a namespaced function answered as a custom call refused
        expect_400(R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                       R"JX({"type":"custom_tool_call","namespace":"multi_agent_v1","name":"spawn_agent","call_id":"c1","input":"x"},)JX"
                       + out1.substr(0, out1.size() - 1) + "]," + tools + "}",
                   "a custom_tool_call carrying a namespace refused (custom tools are never namespaced)");
        expect_400(R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                       R"JX({"type":"function_call","namespace":"multi_agent_v1","name":"spawn_agent","call_id":"c1","arguments":"{}"},)JX"
                       + out1.substr(0, out1.size() - 1) + R"JX(],"tools":[{"type":"custom","name":"multi_agent_v1/spawn_agent"}]})JX",
                   "a namespaced replay colliding with an offered custom tool of that flat name refused");
        {  // the converse: an offered CUSTOM tool named "spawn_agent" is a DIFFERENT tool - the namespaced
          // replay is not a collision (the reconstruction made the identity explicit), so it replays fine
            ResponsesRequest ok_r = parse(R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user + ns_call +
                                         out1.substr(0, out1.size() - 1) +
                                         R"JX(],"tools":[{"type":"custom","name":"spawn_agent"}]})JX");
            const json::Value *c = call0(ok_r, 1);
            ok(c && c->find("function")->find("name")->as_string("name") == "multi_agent_v1/spawn_agent",
               "a plain-name custom tool does not shadow the namespaced replay identity");
        }
    }
    {  // malformed namespaces are named 400s, never a silent drop of the prefix
        expect_400(R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                       R"JX({"type":"function_call","namespace":7,"name":"f","call_id":"c1","arguments":"{}"},)JX" + out1.substr(0, out1.size() - 1) +
                       "]}",
                   "a non-string namespace refused");
        expect_400(R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                       R"JX({"type":"function_call","namespace":"bad ns","name":"f","call_id":"c1","arguments":"{}"},)JX"
                       + out1.substr(0, out1.size() - 1) + "]}",
                   "a namespace with a space refused (template markup safety)");
        expect_400(R"JX({"model":"qwen3.8-flash-next","input":[)JX" + user +
                       R"JX({"type":"function_call","namespace":"a<b","name":"f","call_id":"c1","arguments":"{}"},)JX"
                       + out1.substr(0, out1.size() - 1) + "]}",
                   "a namespace with '<' refused");
    }
}


// Codex view_image returns a content array with an octet-stream data URL.
void check_image_output() {
    const std::string png = "data:application/octet-stream;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p1sAAAAASUVORK5CYII=";
    const auto request = [&](const std::string &url) {
        json::Value b = json::Value::parse(head(
            R"({"role":"user","content":"look"},{"type":"function_call","name":"view_image","call_id":"img","arguments":"{}"},)"
            R"({"type":"function_call_output","call_id":"img","output":[]},{"role":"user","content":"describe"})"));
        json::Value parts = json::Value::array();
        for (const char *text : {"before", "after"}) {
            if (std::string(text) == "after") {
                json::Value part = json::Value::object();
                part.set("type", json::Value::string("input_image"));
                part.set("image_url", json::Value::string(url));
                part.set("detail", json::Value::string("high"));
                parts.push(part);
            }
            json::Value part = json::Value::object();
            part.set("type", json::Value::string("input_text"));
            part.set("text", json::Value::string(text));
            parts.push(part);
        }
        auto items = b.find("input")->as_array("input");
        items[2].set("output", parts);
        json::Value input = json::Value::array();
        for (auto &item : items) input.push(item);
        b.set("input", input);
        return b;
    };
    const json::Value b = request(png);
    try {
        auto r = parse_responses_request(b, kDefaults, kModel, true);
        const auto &parts = at(r, 2).find("content")->as_array("parts");
        ok(parts.size() == 3 && parts[0].find("text")->as_string("text") == "before" &&
           parts[1].find("type")->as_string("type") == "image_url" &&
           parts[2].find("text")->as_string("text") == "after", "tool output text/image order preserved");
        ok(r.chat.images.size() == 1, "one tool output image reaches vision pipeline");
        const auto prompt = render_chat(r.chat.messages, nullptr, r.chat.template_options, nullptr, true);
        ok(prompt.find("before") < prompt.find("after"), "image tool output renders in order");
        auto images = vision::prepare_images(r.chat.images, vision::ServedVision{});
        ok(images.size() == 1 && images[0].merged_tokens > 0, "octet-stream PNG decodes with existing limits");
    } catch (const std::exception &e) { ok(false, "Codex image replay rejected: ", e.what()); }
    expect_400(b.dump(), "image tool output refuses without vision");
    for (const std::string &url : {std::string("https://example.com/image.png"),
                                  std::string("data:application/octet-stream;base64,aGk=")}) {
        try {
            parse_responses_request(request(url), kDefaults, kModel, true);
            ok(false, "unsupported tool image accepted: ", url);
        } catch (const ApiError &e) { ok(e.status == 400 && e.param == "input", "invalid image named 400"); }
    }
}

}  // namespace

int main(int argc, char **argv) {
    const std::string path = argc > 1 ? argv[1] : "tools/testdata/codex-history-2turn.json";
    try {
        check_image_output();
        check_single_call();
        check_parallel_and_delayed();
        check_custom_round();
        check_pairing_rules();
        check_historical_tool_absent();
        check_reasoning_metadata();
        check_namespaced_history();
        std::ifstream in(path);
        STRIX_CHECK(in.is_open(), "cannot read the history fixture '", path, "'");
        std::ostringstream ss;
        ss << in.rdbuf();
        json::Value fixture = json::Value::parse(ss.str());
        json::Value clean = json::Value::object();
        for (const auto &kv : fixture.as_object("fixture"))
            if (!(kv.first.empty() || kv.first[0] == '_')) clean.set(kv.first, kv.second);
        check_two_turn_fixture(clean);
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
