// The /v1/responses tool registry, host only (STORY-04 / gate B4): the real captured Codex function tools
// translate to the nested Chat schema without losing a field, the registry gives the stable external <-> model
// name mapping (namespaced tools included), the mapping is deterministic, the no-tools and tool_choice:none paths
// keep the generator's view empty, and name/schema abuse is refused. No model, no tokenizer.
//
//   check_responses_tools [codex-request-sample.json]   (default: tools/testdata/codex-request-sample.json)

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

const SamplingParams kDefaults;
const std::string kModel = "qwen3.8-flash-next";

json::Value load_fixture() {
    std::ifstream in("tools/testdata/codex-request-sample.json");
    STRIX_CHECK(in.is_open(), "cannot read the Codex request fixture");
    std::ostringstream ss;
    ss << in.rdbuf();
    json::Value v = json::Value::parse(ss.str());
    json::Value clean = json::Value::object();
    for (const auto &kv : v.as_object("fixture"))
        if (!(kv.first.empty() || kv.first[0] == '_' || kv.first == "codex_version")) clean.set(kv.first, kv.second);
    return clean;
}

// The registry entry for a model-facing (flat, external) tool name, or nullptr.
const json::Value *entry(const ResponsesRequest &r, const std::string &external) {
    for (const auto &e : r.tool_registry.as_array("registry"))
        if (e.find("external")->as_string("external") == external) return &e;
    return nullptr;
}

// The nested Chat tool form for a model-facing name, or nullptr.
const json::Value *chat_tool(const ResponsesRequest &r, const std::string &name) {
    for (const auto &t : r.chat.tools.as_array("tools"))
        if (t.find("function")->find("name")->as_string("name") == name) return &t;
    return nullptr;
}

// --- the real captured Codex tools: schema fidelity, field by field ------------------------------------------

void check_fixture_round_trip(const json::Value &fixture) {
    ResponsesRequest r = parse_responses_request(fixture, kDefaults, kModel);

    // Every model-callable tool of the capture, flat names in request order (4 functions, then the namespace's 5).
    const char *want_order[] = {"exec_command", "write_stdin", "request_user_input", "view_image",
                                "multi_agent_v1/close_agent", "multi_agent_v1/resume_agent",
                                "multi_agent_v1/send_input", "multi_agent_v1/spawn_agent", "multi_agent_v1/wait_agent"};
    const auto &reg = r.tool_registry.as_array("registry");
    ok(reg.size() == 9, "registry size, got ", reg.size());
    bool order_ok = reg.size() == 9;
    for (size_t k = 0; k < reg.size() && k < 9; ++k)
        order_ok = order_ok && reg[k].find("external")->as_string("external") == want_order[k];
    ok(order_ok, "registry order mirrors the request order");

    // Field-by-field fidelity against the fixture's own tool entries (the schema objects are compared as the
    // parser received them: key order included, so a re-ordering would show).
    size_t checked = 0;
    for (const auto &t : fixture.find("tools")->as_array("tools")) {
        const std::string &type = t.find("type")->as_string("type");
        if (type == "function") {
            const std::string &name = t.find("name")->as_string("name");
            const json::Value *ct = chat_tool(r, name);
            ok(ct != nullptr, "fixture tool '", name, "' reached the nested Chat form");
            if (!ct) continue;
            const json::Value &fn = *ct->find("function");
            ok(fn.find("parameters")->dump() == t.find("parameters")->dump(),
               "parameters schema of '", name, "' is byte-identical (all properties, enums, nested items)");
            ok(fn.find("description")->as_string("description") == t.find("description")->as_string("description"),
               "description of '", name, "' survives verbatim");
            ok(fn.find("strict") && fn.find("strict")->as_bool("strict") == t.find("strict")->as_bool("strict"),
               "strict flag of '", name, "' preserved");
            ok(ct->find("type")->as_string("type") == "function", "Chat form keeps type=function");
            ++checked;
        } else if (type == "namespace") {
            const std::string ns = t.find("name")->as_string("name");
            for (const auto &c : t.find("tools")->as_array("tools")) {
                const std::string ext = ns + "/" + c.find("name")->as_string("name");
                const json::Value *ct = chat_tool(r, ext);
                ok(ct != nullptr, "namespaced tool '", ext, "' flattened into the Chat schema");
                if (ct) {
                    const json::Value &fn = *ct->find("function");
                    ok(fn.find("parameters")->dump() == c.find("parameters")->dump(),
                       "schema of '", ext, "' byte-identical");
                    ok(fn.find("description")->as_string("description") ==
                               c.find("description")->as_string("description"),
                       "description of '", ext, "' survives");
                }
                const json::Value *e = entry(r, ext);
                ok(e != nullptr, "registry entry for '", ext, "'");
                if (e) {
                    ok(e->find("kind")->as_string("kind") == "function", "kind of '", ext, "'");
                    ok(e->find("namespace")->as_string("namespace") == ns,
                       "namespace recorded for output restore of '", ext, "'");
                }
                ++checked;
            }
        }
    }
    ok(checked == 9, "every callable tool of the capture was checked, got ", checked);

    // The exec_command schema specifics that later stories depend on (the tool that drives the Codex loop).
    const json::Value *exec = chat_tool(r, "exec_command");
    ok(exec != nullptr, "exec_command present");
    if (exec) {
        const json::Value &props = *exec->find("function")->find("parameters")->find("properties");
        for (const char *prop : {"cmd", "justification", "login", "max_output_tokens", "prefix_rule",
                                 "sandbox_permissions", "shell", "tty", "workdir"})
            ok(props.find(prop) != nullptr, "exec_command property '", prop, "' present");
        ok(props.find("sandbox_permissions")->find("enum")->as_array("enum").size() == 2,
           "exec_command sandbox_permissions enum intact");
        ok(exec->find("function")->find("parameters")->find("required") != nullptr,
           "exec_command required list intact");
    }

    // The hosted web_search tool: no entry, no Chat tool (the generator cannot call what it never sees).
    ok(entry(r, "web_search") == nullptr, "hosted web_search not registered");
    ok(chat_tool(r, "web_search") == nullptr, "hosted web_search not in the Chat schema");

    // Registry entries carry exactly the mapping fields (the serializer restores items from these alone).
    bool shape_ok = true;
    for (const auto &e : reg) {
        const auto &obj = e.as_object("entry");
        if (obj.size() != 3) shape_ok = false;
        for (const auto &kv : obj)
            if (kv.first != "external" && kv.first != "kind" && kv.first != "namespace") shape_ok = false;
    }
    ok(shape_ok, "registry entries carry exactly {external, kind, namespace}");

    // Determinism: parse the same capture twice - the model sees byte-identical tool definitions and mapping.
    ResponsesRequest r2 = parse_responses_request(fixture, kDefaults, kModel);
    ok(r.chat.tools.dump() == r2.chat.tools.dump(), "Chat tool schemas deterministic");
    ok(r.tool_registry.dump() == r2.tool_registry.dump(), "registry deterministic");
}

// --- registry rules on varied shapes -----------------------------------------------------------------------

struct Parsed {
    ResponsesRequest r;
};

Parsed parse_tools(const std::string &tools_json, const std::string &tool_choice = "auto") {
    Parsed p;
    json::Value b = json::Value::parse("{\"model\":\"" + kModel + "\",\"input\":\"q\",\"tools\":" + tools_json +
                                       ",\"tool_choice\":\"" + tool_choice + "\"}");
    p.r = parse_responses_request(b, kDefaults, kModel);
    return p;
}

void expect_tools_400(const std::string &tools_json, const std::string &what) {
    try {
        parse_tools(tools_json);
        ok(false, what, ": accepted, expected 400 param tools");
    } catch (const ApiError &e) {
        ok(e.status == 400 && e.param == "tools", what, ": got ", e.status, " param '", e.param, "'");
    }
}

void check_registry_rules() {
    // Tools absent: the generator sees no tool definitions at all.
    {
        json::Value b = json::Value::parse("{\"model\":\"" + kModel + "\",\"input\":\"q\"}");
        ResponsesRequest r = parse_responses_request(b, kDefaults, kModel);
        ok(r.chat.tools.is_null(), "no tools -> the model sees no tool definitions");
        ok(r.tool_registry.as_array("reg").empty(), "no tools -> empty registry");
    }
    // Empty array: same.
    ok(parse_tools("[]").r.chat.tools.is_null(), "tools:[] -> null Chat tools");

    // tool_choice none with real tools: generator sees none, registry stays (output mapping survives).
    {
        Parsed p = parse_tools("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]", "none");
        ok(p.r.chat.tools.is_null(), "tool_choice none -> no tools to the generator");
        ok(p.r.tool_registry.as_array("reg").size() == 1, "tool_choice none -> registry kept");
    }
    // tool_choice auto: tools present.
    ok(!parse_tools("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]").r.chat.tools.is_null(),
       "tool_choice auto -> tools reach the generator");

    // A namespace child keeping its description AND the model-facing flat name (round trip through the registry).
    {
        Parsed p = parse_tools("[{\"type\":\"namespace\",\"name\":\"multi_agent_v1\",\"tools\":"
                               "[{\"type\":\"function\",\"name\":\"spawn_agent\",\"description\":\"spawn\","
                               "\"parameters\":{\"type\":\"object\",\"properties\":{\"task\":{\"type\":\"string\"}}}}]}]");
        const json::Value *ct = chat_tool(p.r, "multi_agent_v1/spawn_agent");
        ok(ct != nullptr, "flat model-facing name for the namespace child");
        ok(ct && ct->find("function")->find("description")->as_string("d") == "spawn", "child description kept");
        const json::Value *e = entry(p.r, "multi_agent_v1/spawn_agent");
        ok(e && e->find("namespace")->as_string("namespace") == "multi_agent_v1" &&
                   e->find("kind")->as_string("kind") == "function",
           "registry restores namespace + tool-call type on output");
    }

    // Names that would forge the model's call tags (the parser reads <function=NAME> back) are refused.
    expect_tools_400("[{\"type\":\"function\",\"name\":\"a>b\",\"parameters\":{\"type\":\"object\"}}]", "name with '>'");
    expect_tools_400("[{\"type\":\"function\",\"name\":\"a<b\",\"parameters\":{\"type\":\"object\"}}]", "name with '<'");
    expect_tools_400("[{\"type\":\"function\",\"name\":\"a=b\",\"parameters\":{\"type\":\"object\"}}]", "name with '='");
    // A name of 256 bytes is the documented bound; 257 is refused.
    {
        const std::string n256(256, 'x'), n257(257, 'x');
        Parsed p = parse_tools("[{\"type\":\"function\",\"name\":\"" + n256 + "\",\"parameters\":{\"type\":\"object\"}}]");
        ok(entry(p.r, n256) != nullptr, "256-byte name accepted");
        expect_tools_400("[{\"type\":\"function\",\"name\":\"" + n257 + "\",\"parameters\":{\"type\":\"object\"}}]",
                         "257-byte name");
    }
    // The schema's top level must describe an object (the parser types arguments from its properties).
    expect_tools_400("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"string\"}}]",
                     "parameters of non-object type");
    expect_tools_400("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{}}]", "parameters without type");
    // Collision between a plain and a namespaced external name: refused, deterministically, on every parse.
    expect_tools_400("[{\"type\":\"function\",\"name\":\"ns/f\",\"parameters\":{\"type\":\"object\"}},"
                     "{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"f\","
                     "\"parameters\":{\"type\":\"object\"}}]}]", "ns/f collision (order 1)");
    expect_tools_400("[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"f\","
                     "\"parameters\":{\"type\":\"object\"}}]},{\"type\":\"function\",\"name\":\"ns/f\","
                     "\"parameters\":{\"type\":\"object\"}}]", "ns/f collision (order 2)");
    // Description/strict optionality: accepted without, kept with.
    {
        Parsed p = parse_tools("[{\"type\":\"function\",\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}]");
        const json::Value *ct = chat_tool(p.r, "f");
        ok(ct && ct->find("function")->find("description") == nullptr, "no description -> none invented");
        ok(ct && ct->find("function")->find("strict") == nullptr, "no strict -> none invented");
    }
    // 128 callable tools is the bound; 129 is refused.
    {
        std::string many = "[";
        for (int k = 0; k < 128; ++k)
            many += (k ? "," : "") + std::string("{\"type\":\"function\",\"name\":\"f") + std::to_string(k) +
                    "\",\"parameters\":{\"type\":\"object\"}}";
        many += "]";
        ok(parse_tools(many).r.tool_registry.as_array("reg").size() == 128, "128 tools accepted");
        std::string over = many.substr(0, many.size() - 1) +
                           ",{\"type\":\"function\",\"name\":\"f128\",\"parameters\":{\"type\":\"object\"}}]";
        expect_tools_400(over, "129 tools");
    }
}

}  // namespace

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    try {
        check_fixture_round_trip(load_fixture());
        check_registry_rules();
    } catch (const std::exception &e) {
        std::printf("FAIL (unhandled): %s\n", e.what());
        return 2;
    }
    std::printf("%s: %d/%d checks passed\n", failures == 0 ? "PASS" : "FAILED", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
