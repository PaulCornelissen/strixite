// The /v1/responses request contract (for Codex), host only: normalizing real captured Codex requests and their
// negative cases through parse_responses_request, before any generation. No model, no GPU, no tokenizer.json:
// the checks run on JSON in, JSON out. The fixture tools/testdata/codex-request-sample.json is a sanitized request
// captured from codex-cli 0.162.1 (scripts/record_codex.py in the design pack); the inline cases vary it.
//
//   check_responses [codex-request-sample.json]     (default: tools/testdata/codex-request-sample.json)

#include "common/check.hpp"
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

const SamplingParams kDefaults;  // generation_config defaults are not the point of these tests
const std::string kModel = "qwen3.8-flash-next";

// The served-model gate: the Responses route must validate the name, not serve any model as the Chat route does.
json::Value load_fixture(const std::string &path) {
    std::ifstream in(path);
    STRIX_CHECK(in.is_open(), "cannot read the Codex request fixture '", path, "'");
    std::ostringstream ss;
    ss << in.rdbuf();
    json::Value v = json::Value::parse(ss.str());
    STRIX_CHECK(v.is_object() && v.find("input") != nullptr, "fixture is not a Responses request body");
    // Drop the fixture's own provenance keys: they are test metadata, not request fields, and the strict
    // top-level key check of parse_responses_request must see exactly what Codex sent.
    json::Value clean = json::Value::object();
    for (const auto &kv : v.as_object("fixture"))
        if (!(kv.first.empty() || kv.first[0] == '_' || kv.first == "codex_version")) clean.set(kv.first, kv.second);
    return clean;
}

const json::Value *field(const json::Value &obj, const char *name) { return obj.find(name); }

// One entry of the parsed request's tool registry, by the external (client-facing, flat) tool name.
const json::Value *registry_entry(const ResponsesRequest &r, const std::string &external) {
    for (const auto &e : r.tool_registry.as_array("tool_registry"))
        if (e.find("external")->as_string("external") == external) return &e;
    return nullptr;
}

// --- the captured Codex request, as sent ---------------------------------------------------------------

void check_real_codex_request(const json::Value &fixture) {
    ResponsesRequest r = parse_responses_request(fixture, kDefaults, kModel);
    ok(r.chat.stream, "the captured request streams");
    ok(r.chat.model == kModel, "the model echoes the served id");

    // input: [developer instructions, developer skills, user environment_context, user prompt] in order. The
    // template's roles are system/user/assistant/tool; developer items normalize to system, and the leading
    // system run merges to ONE (B1: the template accepts one leading system - the Chat route's rule, shared).
    const auto &msgs = r.chat.messages.as_array("chat.messages");
    ok(msgs.size() == 3, "the leading developer pair merges: 4 items -> 3 chat messages, got ", msgs.size());
    ok(msgs[0].find("role")->as_string("role") == "system", "the merged leading system is first");
    const std::string &sys = msgs[0].find("content")->as_string("content");
    ok(sys.find("coding agent") != std::string::npos, "the developer instructions text is kept in the merged system");
    ok(sys.find("<skills_instructions>") != std::string::npos &&
               sys.find("coding agent") < sys.find("<skills_instructions>"),
       "both developer texts merged in their original order (blank-line joined)");
    ok(msgs[1].find("role")->as_string("role") == "user" && msgs[2].find("role")->as_string("role") == "user",
       "user items keep their role and order");
    ok(msgs[2].find("content")->as_string("content").find("one word: hello") != std::string::npos,
       "the user's prompt text reaches the last message intact");

    // tools: 4 functions + 5 namespaced children = 9 model-callable tools; the hosted web_search is not executed
    // by this server and is dropped (disclosed in docs/server.md), never silently passed to the generator.
    ok(r.chat.tools.as_array("chat.tools").size() == 9, "flattened tool count, got ", r.chat.tools.as_array("chat.tools").size());
    ok(registry_entry(r, "web_search") == nullptr, "the hosted web_search tool is not registered");
    const json::Value *exec = registry_entry(r, "exec_command");
    ok(exec != nullptr, "exec_command registered");
    if (exec != nullptr) ok(exec->find("kind")->as_string("kind") == "function", "exec_command kind is function");
    const json::Value *child = registry_entry(r, "multi_agent_v1/close_agent");
    ok(child != nullptr, "namespaced close_agent registered under its namespaced external name");
    if (child != nullptr) {
        ok(child->find("kind")->as_string("kind") == "function", "namespace children are functions");
        ok(child->find("namespace")->as_string("namespace") == "multi_agent_v1",
           "the namespace is recorded for output item restore");
    }

    // The nested Chat schema of exec_command keeps every field Codex sent (STORY-04 acceptance).
    const json::Value *nested = nullptr;
    for (const auto &t : r.chat.tools.as_array("chat.tools"))
        if (t.find("function")->find("name")->as_string("name") == "exec_command") nested = &t;
    ok(nested != nullptr, "exec_command reaches the nested Chat tool form");
    if (nested != nullptr) {
        const json::Value &fn = *nested->find("function");
        ok(fn.find("description")->as_string("description").find("PTY") != std::string::npos,
           "the tool description survives");
        ok(fn.find("parameters")->find("properties")->find("cmd") != nullptr,
           "the parameters schema (cmd) survives");
        ok(fn.find("parameters")->find("properties")->find("sandbox_permissions")->find("enum") != nullptr,
           "deep schema fields (sandbox_permissions.enum) survive");
    }

    // Verified options: benign client metadata (prompt_cache_key, client_metadata, include, parallel_tool_calls)
    // parses; store:false is the stateless mode; tool_choice/reasoning carry through.
    ok(r.tool_choice == "auto", "tool_choice auto");
    ok(r.reasoning_effort == "medium", "reasoning effort medium, got ", r.reasoning_effort);
}

// --- negative cases: path-specific 400s, never 500, never silent ------------------------------------------

void expect_400(const json::Value &body, const std::string &param, const std::string &what) {
    try {
        parse_responses_request(body, kDefaults, kModel);
        ok(false, what, ": expected a 400, parsed fine");
    } catch (const ApiError &e) {
        ok(e.status == 400 && e.param == param, what, ": got status ", e.status, " param '", e.param,
           "' (want 400 / '", param, "')");
    }
}

json::Value with(json::Value base, const char *key, const std::string &raw_value) {
    json::Value v = json::Value::parse(raw_value);
    base.set(key, std::move(v));
    return base;
}

void check_negatives(const json::Value &fixture) {
    expect_400(with(fixture, "model", "\"gpt-6-astra\""), "model", "a model name the server is not");
    expect_400(with(fixture, "store", "true"), "store", "store:true (no response state store here)");
    expect_400(with(fixture, "tool_choice", "\"required\""), "tool_choice",
               "tool_choice:required (not served: the model is never forced)");
    expect_400(with(fixture, "input", "[]"), "input", "an empty input array");
    expect_400(with(fixture, "max_output_tokens", "0"), "max_output_tokens", "max_output_tokens 0");

    json::Value bad_input = json::Value::parse(
        "[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"hi\"}]},"
        "{\"type\":\"item_from_mars\",\"role\":\"user\",\"content\":[]}]");
    expect_400(with(fixture, "input", bad_input.dump()), "input", "an unrecognized input item type");

    json::Value bad_tools = json::Value::parse(
        "[{\"type\":\"function\",\"name\":\"dupe\",\"parameters\":{\"type\":\"object\",\"properties\":{}}},"
        "{\"type\":\"function\",\"name\":\"dupe\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]");
    expect_400(with(fixture, "tools", bad_tools.dump()), "tools", "duplicate tool names");

    json::Value bad_tool = json::Value::parse("[{\"type\":\"function\",\"name\":\"nosc\"}]");
    expect_400(with(fixture, "tools", bad_tool.dump()), "tools", "a function tool without a parameters schema");

    expect_400(with(fixture, "background", "true"), "background", "background:true (no background mode)");

    // benign metadata stays accepted
    ResponsesRequest r = parse_responses_request(with(fixture, "include", "[\"reasoning.encrypted_content\"]"),
                                                 kDefaults, kModel);
    ok(r.chat.stream, "include:[reasoning.encrypted_content] is accepted (raw reasoning; no encrypted content)");

    // tool_choice:none - the generator sees no tools, the registry keeps them for output mapping (STORY-04 rule).
    ResponsesRequest none = parse_responses_request(with(fixture, "tool_choice", "\"none\""), kDefaults, kModel);
    ok(none.chat.tools.is_null(), "tool_choice none: the generator gets no tool definitions");
    ok(none.tool_registry.as_array("tool_registry").size() == 9, "tool_choice none: the registry stays complete");
}


// --- STORY-11: structured output (text.format) and image parts (the vision passthrough rules) ---------------
// text.format lands on the Chat pipeline's response_format fields (the grammar compiles on the route - the
// wire proof is in check_responses_http; here: the normalization). Image parts: data URLs reach the Chat
// image pipeline IN ORDER when the server has a tower; everything else (no tower, remote URLs, file ids,
// non-user content, audio/video) is a named refusal - never a silent drop, and the server fetches nothing.

std::string err_param_of(const std::string &body, int *status, std::string *msg, bool vision = false) {
    try {
        parse_responses_request(json::Value::parse(body), kDefaults, kModel, vision);
        if (status) *status = 0;
        return {};
    } catch (const ApiError &e) {
        if (status) *status = e.status;
        if (msg) *msg = e.what();
        return e.param;
    } catch (const Error &e) {
        ok(false, "test body is not valid JSON: ", body);
        if (status) *status = -1;
        return "BAD-BODY";
    }
}

ResponsesRequest pres(const std::string &body, bool vision = false) {
    json::Value v;
    try {
        v = json::Value::parse(body);
    } catch (const Error &e) {
        ok(false, "test body is not valid JSON (byte ", ((const std::string &)e.what()).size(), "): ", body);
        throw;
    }
    return parse_responses_request(v, kDefaults, kModel, vision);
}

void check_structured_and_images() {
    const std::string m = kModel;
    const std::string pre = "{\"model\":\"" + m + "\",\"input\":\"q\"";
    {  // the format table -> the Chat fields (the same compile site, the same enforcement)
        ok(pres(pre + ",\"text\":{\"format\":{\"type\":\"text\"}}}").chat.response_format ==
               ChatRequest::ResponseFormat::Text,
           "format text stays Text");
        ok(pres(pre + ",\"text\":{\"format\":{\"type\":\"json_object\"}}}").chat.response_format ==
               ChatRequest::ResponseFormat::JsonObject,
           "json_object maps to the Chat json_object grammar mode");
        ResponsesRequest r = pres(pre + ",\"text\":{\"format\":{\"type\":\"json_schema\",\"name\":\"ans\","
                                             "\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":"
                                             "{\"answer\":{\"type\":\"integer\"}},\"required\":[\"answer\"]}}}}");
        ok(r.chat.response_format == ChatRequest::ResponseFormat::JsonSchema, "json_schema maps");
        ok(r.chat.response_schema_name == "ans", "the schema name carries over");
        ok(r.chat.response_schema.find("properties")->find("answer")->find("type")->as_string("t") == "integer",
           "the schema object carries over intact");
    }
    {  // format-shape refusals (param text)
        int st = 0;
        std::string msg;
        ok(err_param_of(pre + ",\"text\":{\"format\":{\"type\":\"json_schema\"}}}", &st, &msg) == "text" && st == 400,
           "json_schema without a schema -> 400 text");
        ok(err_param_of(pre + ",\"text\":{\"format\":{\"type\":\"json_schema\",\"schema\":[1]}}}", &st, &msg) ==
                   "text" &&
               st == 400,
           "a non-object schema -> 400 text");
        ok(err_param_of(pre + ",\"text\":{\"format\":{\"type\":\"json_schema\",\"schema\":{},\"strict\":\"yes\"}}}",
                        &st, &msg) == "text" &&
                   st == 400,
           "strict must be a bool");
        ok(err_param_of(pre + ",\"text\":{\"format\":{\"type\":\"json_schema\",\"schema\":{},\"name\":42}}}", &st,
                        &msg) == "text" &&
                   st == 400,
           "name must be a string");
        std::string p = err_param_of(pre + ",\"text\":{\"format\":{\"type\":\"grammar\"}}}", &st, &msg);
        ok(p == "text" && st == 400 && msg.find("json_object") != std::string::npos,
           "an unknown format type names the served set, got: ", msg);
        ok(err_param_of(pre + ",\"text\":{\"format\":\"text\"}}", &st, &msg) == "text" && st == 400,
           "format must be an object");
    }
    // --- images -------------------------------------------------------------------------------
    const char *kData = "data:image/png;base64,aGk=";   // the bytes "hi"
    const char *kData2 = "data:image/png;base64,YWJj";  // the bytes "abc"
    {
        int st = 0;
        std::string msg;
        std::string p = err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":["
                                     "{\"type\":\"input_text\",\"text\":\"look\"},{\"type\":\"input_image\",\"image_url\":\"" +
                                         std::string(kData) + "\"}]}]}",
                                     &st, &msg, /*vision=*/false);
        ok(p == "input" && st == 400 && msg.find("vision tower") != std::string::npos,
           "no tower: an image part is a named refusal (never a silent drop), got: ", msg);
    }
    {
        ResponsesRequest r = pres("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":["
                                 "{\"type\":\"input_text\",\"text\":\"look\"},{\"type\":\"input_image\",\"image_url\":\"" +
                                     std::string(kData) + "\",\"detail\":\"high\"}]}]}",
                                 /*vision=*/true);
        ok(r.chat.images.size() == 1, "the image reached the Chat pipeline");
        ok(r.chat.images[0].bytes == std::vector<uint8_t>({'h', 'i'}), "the data URL decoded (bytes 'hi')");
        ok(r.chat.images[0].mime == "image/png", "the mime carried over");
        const json::Value &msg = r.chat.messages.as_array("messages")[0];
        ok(msg.find("content")->is_array(), "an image message keeps its parts (array content)");
        const std::vector<json::Value> &parts = msg.find("content")->as_array("content");
        ok(parts.size() == 2 && parts[0].find("type")->as_string("t") == "text" &&
                   parts[1].find("type")->as_string("t") == "image_url" &&
                   parts[1].find("image_url")->find("url")->as_string("u") == kData,
           "text then image, the data URL intact (the Chat shape the template renders)");
    }
    {  // ordering: text, image, text inside one item, then a second item with two images
        ResponsesRequest r = pres("{\"model\":\"" + m + "\",\"input\":["
                                 "{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"a\"},"
                                 "{\"type\":\"input_image\",\"image_url\":\"" + std::string(kData) + "\"},"
                                 "{\"type\":\"input_text\",\"text\":\"b\"}]},"
                                 "{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":"
                                 "{\"url\":\"" + std::string(kData2) + "\"}}]}]}",
                                 true);
        ok(r.chat.images.size() == 2, "both images collected, document order");
        ok(r.chat.images[0].bytes == std::vector<uint8_t>({'h', 'i'}) &&
                   r.chat.images[1].bytes == std::vector<uint8_t>({'a', 'b', 'c'}),
           "images ordered by (item, part): hi then abc (the lax image_url.url shape accepted)");
        const std::vector<json::Value> &parts =
            r.chat.messages.as_array("messages")[0].find("content")->as_array("content");
        ok(parts.size() == 3 && parts[1].find("type")->as_string("t") == "image_url",
           "the interleaving text-image-text is preserved as parts");
    }
    {
        int st = 0;
        std::string msg;
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
                        "\"image_url\":\"https://example.com/a.png\"}]}]}",
                        &st, &msg, true) == "input" &&
                   st == 400 && msg.find("data:") != std::string::npos &&
                   msg.find("fetches nothing") != std::string::npos,
           "remote URLs refused (the server fetches nothing), got: ", msg);
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
                        "\"file_id\":\"file_1\"}]}]}",
                        &st, &msg, true) == "input" &&
                   st == 400 && msg.find("file_id") != std::string::npos,
           "file ids refused (stateless: no store)");
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
                        "\"image_url\":\"data:image/png;base64,zz!!\"}]}]}",
                        &st, &msg, true) == "input" &&
                   st == 400,
           "a malformed base64 data URL refused");
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\","
                        "\"image_url\":\"" +
                            std::string(kData) + "\",\"detail\":\"mega\"}]}]}",
                        &st, &msg, true) == "input" &&
                   st == 400 && msg.find("detail") != std::string::npos,
           "detail validated");
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"assistant\",\"content\":[{\"type\":\"input_image\","
                        "\"image_url\":\"" +
                            std::string(kData) + "\"}]}]}",
                        &st, &msg, true) == "input" &&
                   st == 400 && msg.find("user") != std::string::npos,
           "images belong to user messages");
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":\"x\"},"
                        "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                        "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":[{\"type\":\"input_image\","
                        "\"image_url\":\"" +
                            std::string(kData) + "\"}]}]}",
                        &st, &msg, false) == "input" &&
                   st == 400 && msg.find("vision tower") != std::string::npos,
           "tool output images refuse explicitly when vision is disabled");
        ok(err_param_of("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\","
                        "\"input_audio\":{}}]}]}",
                        &st, &msg, true) == "input" &&
                   st == 400 && msg.find("audio") != std::string::npos,
           "audio refused, named");
    }
    {  // parity: the same data URL through the Chat parser and this one decodes identically
        ResponsesRequest rr = pres("{\"model\":\"" + m + "\",\"input\":[{\"role\":\"user\",\"content\":["
                                  "{\"type\":\"input_image\",\"image_url\":\"" +
                                      std::string(kData2) + "\"}]}]}",
                                  true);
        ChatRequest cr = parse_chat_request(json::Value::parse("{\"messages\":[{\"role\":\"user\",\"content\":["
                                                               "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" +
                                                               std::string(kData2) + "\"}}]}]}"),
                                            kDefaults, true);
        ok(cr.images.size() == 1 && rr.chat.images.size() == 1 && cr.images[0].bytes == rr.chat.images[0].bytes &&
                   cr.images[0].mime == rr.chat.images[0].mime,
           "one data-URL rule serves both routes (byte-for-byte parity with the Chat parser)");
    }
}


// --- external review B1 (2026-10-27): leading developer/system messages render as ONE system, a
// mid-conversation one is a named 400. Codex's real captured request opens with TWO developer messages -
// the template accepts a single leading system, so parse-only proof hid a request that fails at render.

void check_leading_system() {
    const std::string m = kModel;
    {  // the fixture shape: two leading developer messages + user -> ONE merged system (the Chat rule)
        ResponsesRequest r = pres("{\"model\":\"" + m + "\",\"input\":["
                                 "{\"role\":\"developer\",\"content\":\"base instructions\"},"
                                 "{\"role\":\"developer\",\"content\":\"<skills/> guidance\"},"
                                 "{\"role\":\"user\",\"content\":\"go\"}]}");
        const auto &ms = r.chat.messages.as_array("messages");
        ok(ms.size() == 2, "two leading developer messages merge into system + user");
        ok(!ms.empty() && ms[0].find("role")->as_string("r") == "system" &&
                   !ms.empty() && ms[0].find("content")->as_string("c") == "base instructions\n\n<skills/> guidance",
           "merged in order with the Chat delimiter (blank line)");
    }
    {  // instructions joins the leading system group too
        ResponsesRequest r = pres("{\"model\":\"" + m + "\",\"instructions\":\"instr\",\"input\":["
                                 "{\"role\":\"system\",\"content\":\"s1\"},{\"role\":\"user\",\"content\":\"go\"}]}");
        const auto &ms = r.chat.messages.as_array("messages");
        ok(ms.size() == 2 && ms[0].find("content")->as_string("c") == "instr\n\ns1",
           "instructions + a leading system merge in order");
    }
    {  // mid-conversation system/developer: a typed 400 naming the field, not a render-time failure
        int st = 0;
        std::string msg;
        std::string p1 = err_param_of("{\"model\":\"" + m + "\",\"input\":["
                                      "{\"role\":\"user\",\"content\":\"go\"},{\"role\":\"developer\",\"content\":\"mid\"}]}",
                                      &st, &msg);
        ok(p1 == "input" && st == 400 && msg.find("leading") != std::string::npos,
           "a system/developer item after the conversation started is a named 400, got: ", msg);
        std::string p2 = err_param_of("{\"model\":\"" + m + "\",\"input\":["
                                      "{\"role\":\"user\",\"content\":\"go\"},"
                                      "{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"f\",\"arguments\":\"{}\"},"
                                      "{\"type\":\"function_call_output\",\"call_id\":\"c1\",\"output\":\"ok\"},"
                                      "{\"role\":\"system\",\"content\":\"mid\"}]}",
                                      &st, &msg);
        ok(p2 == "input" && st == 400, "a system item after a tool round is the same refusal, got ", st);
    }
    {  // leading run then the conversation: the merged system is index 0 and the order of texts is exact
        ResponsesRequest r = pres("{\"model\":\"" + m + "\",\"input\":["
                                 "{\"role\":\"system\",\"content\":\"a\"},{\"role\":\"developer\",\"content\":\"b\"},"
                                 "{\"role\":\"system\",\"content\":\"c\"},{\"role\":\"user\",\"content\":\"go\"}]}");
        const auto &ms = r.chat.messages.as_array("messages");
        ok(ms.size() == 2 && ms[0].find("content")->as_string("c") == "a\n\nb\n\nc",
           "the whole leading run (any system/developer mix) merges, in order");
    }
}

}  // namespace

int main(int argc, char **argv) {
    const std::string path = argc > 1 ? argv[1] : "tools/testdata/codex-request-sample.json";
    try {
        json::Value fixture = load_fixture(path);
        check_real_codex_request(fixture);
        check_negatives(fixture);
        check_structured_and_images();
        check_leading_system();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
