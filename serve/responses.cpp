#include "serve/responses.hpp"

#include <cctype>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace strix {
namespace {

[[noreturn]] void bad(const std::string &param, const std::string &msg) {
    throw ApiError(400, msg, param);
}

const json::Value *field(const json::Value &o, const char *key) { return o.find(key); }

bool is_string(const json::Value *v) { return v && v->is_string(); }

std::string get_string(const json::Value &v, const std::string &param, const std::string &path) {
    if (!v.is_string()) bad(param, "'" + path + "' must be a string, got " + std::string(json::type_name(v.type())));
    return v.as_string(path);
}

bool get_bool(const json::Value &v, const std::string &param, const std::string &path) {
    if (!v.is_bool()) bad(param, "'" + path + "' must be a boolean, got " + std::string(json::type_name(v.type())));
    return v.as_bool(path);
}

int64_t get_int(const json::Value &v, const std::string &param, const std::string &path) {
    if (!v.is_int()) bad(param, "'" + path + "' must be an integer, got " + std::string(json::type_name(v.type())));
    return v.as_int(path);
}

double get_number(const json::Value &v, const std::string &param, const std::string &path) {
    if (!v.is_number()) bad(param, "'" + path + "' must be a number, got " + std::string(json::type_name(v.type())));
    return v.as_double(path);
}

// A tool's name: non-empty, printable, bounded, and safe for the model's call tags. (Names reach the chat
// template's tool header and the model's <function=NAME> tags; whitespace, control bytes and newlines would forge
// turn structure, and '<', '>' or '=' would forge the call tags the output parser reads back.)
void check_tool_name(const std::string &name, const std::string &param) {
    if (name.empty()) bad(param, "a tool name must be non-empty");
    if (name.size() > 256) bad(param, "tool name '" + name.substr(0, 32) + "...' is longer than 256 bytes");
    for (unsigned char c : name)
        if (c <= 0x20 || c == 0x7f || (c >= 0x80 && c < 0xa0) || c == '<' || c == '>' || c == '=')
            bad(param, "tool name '" + name + "' holds whitespace, a control character, a non-printable byte, "
                        "or a tag-significant character ('<', '>', '=')");
}

// One text content part's text; parts the build can't serve yet are explicit 400s (never dropped).
std::string part_text(const json::Value &part, const std::string &param, const std::string &path) {
    if (!part.is_object()) bad(param, "'" + path + "' must be an object, got " + std::string(json::type_name(part.type())));
    const json::Value *type = part.find("type");
    if (!is_string(type)) bad(param, "'" + path + ".type' must be a string");
    const std::string &t = type->as_string(path + ".type");
    if (t == "input_text" || t == "output_text" || t == "text") {
        const json::Value *text = part.find("text");
        if (!is_string(text)) bad(param, "'" + path + ".text' must be a string");
        return text->as_string(path + ".text");
    }
    if (t == "refusal") {  // assistant refusal text: kept as the message's text (the Chat form has no refusal part)
        const json::Value *r = part.find("refusal");
        if (!is_string(r)) bad(param, "'" + path + ".refusal' must be a string");
        return r->as_string(path + ".refusal");
    }
    if (t == "input_image") bad(param, "'" + path + "' is an image part: " +
                                           "image input lands with vision passthrough on this route (see docs/server.md)");
    if (t == "input_file" || t == "input_audio" || t == "audio" || t == "video")
        bad(param, "'" + path + "' is a '" + t + "' part: only text parts are served");
    bad(param, "'" + path + ".type' is unrecognized: '" + t + "'");
}

// One input item -> a chat message {role, content} (text form). Message items are the STORY-02 surface; the
// history kinds (function_call, tool outputs, reasoning) join at their stories with explicit rejections until then.
json::Value normalize_item(const json::Value &item, size_t k) {
    const std::string path = "input[" + std::to_string(k) + "]";
    if (!item.is_object()) bad("input", "'" + path + "' must be an object, got " + std::string(json::type_name(item.type())));
    const json::Value *type = item.find("type");
    const json::Value *role = item.find("role");
    if (is_string(type) && type->as_string("type") != "message") {
        const std::string &t = type->as_string("type");
        if (t == "function_call" || t == "custom_tool_call")
            bad("input", "'" + path + "' is a '" + t + "' item: tool calls reach history replay (docs/server.md)");
        if (t == "function_call_output" || t == "custom_tool_call_output" || t == "function_output")
            bad("input", "'" + path + "' is a tool-output item: tool results reach history replay (docs/server.md)");
        if (t == "reasoning")
            bad("input", "'" + path + "' is a reasoning item: raw reasoning reaches history replay (docs/server.md)");
        bad("input", "'" + path + ".type' is unrecognized: '" + t + "'");
    }
    if (!is_string(role))
        bad("input", "'" + path + "' needs a string 'role' (message item)" +
                         (is_string(type) ? "" : " and a 'type' of \"message\", an object, or null"));
    const std::string &r = role->as_string(path + ".role");
    std::string chat_role;
    if (r == "user" || r == "assistant") chat_role = r;
    else if (r == "system" || r == "developer") chat_role = "system";  // the template's roles; developer is system-priority text
    else if (r == "tool") bad("input", "'" + path + "' has role 'tool': tool results arrive as *_output items (docs/server.md)");
    else bad("input", "'" + path + ".role' must be system, developer, user, assistant, or tool, got '" + r + "'");

    std::string text;
    const json::Value *content = item.find("content");
    if (!content || content->is_null()) {
        if (chat_role == "assistant") text = "";  // an assistant item with no content: the template's empty answer
        else bad("input", "'" + path + "' (a " + r + " message) needs 'content'");
    } else if (content->is_string()) {
        text = content->as_string(path + ".content");
    } else if (content->is_array()) {
        for (size_t p = 0; p < content->as_array("content").size(); ++p) text += part_text(content->as_array("content")[p], "input", path + ".content[" + std::to_string(p) + "]");
    } else {
        bad("input", "'" + path + ".content' must be a string, an array of content parts, or null, got " +
                         std::string(json::type_name(content->type())));
    }
    json::Value msg = json::Value::object();
    msg.set("role", json::Value::string(chat_role));
    msg.set("content", json::Value::string(std::move(text)));
    return msg;
}

// One Responses function tool -> the nested Chat form {type, function {name, description?, parameters, strict?}}.
// The schema fields ride along untouched (the template renders them; the parser types arguments from them).
void add_function_tool(const json::Value &tool, const std::string &ns, const std::string &external,
                       json::Value &chat_tools, json::Value &registry, std::set<std::string> &names,
                       const std::string &param) {
    const json::Value *name = tool.find("name");
    if (!is_string(name)) bad(param, "a tool needs a string 'name'");
    const std::string &n = name->as_string("tools[].name");
    check_tool_name(n, param);
    const json::Value *params = tool.find("parameters");
    if (!params || !params->is_object())
        bad(param, "tool '" + n + "' needs an object 'parameters' (the JSON schema of its arguments)");
    // The argument schema is an object schema: the output parser types parameter values from its `properties`,
    // and a schema the generator cannot answer with an object is not a callable tool.
    const json::Value *ptype = params->find("type");
    if (!(is_string(ptype) && ptype->as_string("tools[].parameters.type") == "object"))
        bad(param, "tool '" + n + "' parameters: the schema's top-level type must be \"object\", got " +
                       (is_string(ptype) ? "'" + ptype->as_string("tools[].parameters.type") + "'" : "no type"));
    if (!names.insert(external).second) bad(param, "duplicate tool name '" + external + "'");
    json::Value fn = json::Value::object();
    if (const json::Value *d = tool.find("description"); is_string(d))
        fn.set("description", json::Value::string(d->as_string("tools[].description")));
    // The model-facing name is the flat external form (a namespace child as "namespace/name"): what the model
    // emits in <function=NAME> must identify the tool uniquely across namespaces, and it is the registry key.
    fn.set("name", json::Value::string(external));
    fn.set("parameters", *params);
    if (const json::Value *s = tool.find("strict"); s && !s->is_null())
        fn.set("strict", json::Value::boolean(get_bool(*s, param, "tools[].strict")));  // accepted; schemas render as sent
    json::Value entry = json::Value::object();
    entry.set("type", json::Value::string("function"));
    entry.set("function", std::move(fn));
    chat_tools.push(std::move(entry));
    json::Value reg = json::Value::object();
    reg.set("external", json::Value::string(external));
    reg.set("kind", json::Value::string("function"));
    reg.set("namespace", json::Value::string(ns));
    registry.push(std::move(reg));
}

void parse_tools(const json::Value *tools, ResponsesRequest &r, const std::string &tool_choice) {
    json::Value chat_tools = json::Value::array();
    std::set<std::string> names;
    if (tools && !tools->is_null()) {
        if (!tools->is_array()) bad("tools", "'tools' must be an array, got " + std::string(json::type_name(tools->type())));
        const auto &list = tools->as_array("tools");
        for (size_t k = 0; k < list.size(); ++k) {
            const json::Value &t = list[k];
            const std::string path = "tools[" + std::to_string(k) + "]";
            if (!t.is_object()) bad("tools", "'" + path + "' must be an object");
            const json::Value *type = t.find("type");
            if (!is_string(type)) bad("tools", "'" + path + ".type' must be a string");
            const std::string &tt = type->as_string(path + ".type");
            if (tt == "function") {
                const json::Value *nm = t.find("name");
                if (!is_string(nm)) bad("tools", "'" + path + "' needs a string 'name'");
                add_function_tool(t, "", nm->as_string(path + ".name"), chat_tools, r.tool_registry, names, "tools");
            } else if (tt == "namespace") {
                const json::Value *npm = t.find("name");
                if (!is_string(npm)) bad("tools", "'" + path + "' needs a string 'name'");
                const std::string ns = npm->as_string(path + ".name");
                check_tool_name(ns, "tools");
                const json::Value *inner = t.find("tools");
                if (!inner || !inner->is_array() || inner->as_array("tools").empty())
                    bad("tools", "namespace '" + ns + "' needs a non-empty 'tools' array");
                for (size_t j = 0; j < inner->as_array("tools").size(); ++j) {
                    const json::Value &c = inner->as_array("tools")[j];
                    const std::string cpath = path + ".tools[" + std::to_string(j) + "]";
                    if (!c.is_object()) bad("tools", "'" + cpath + "' must be an object");
                    const json::Value *ctype = c.find("type");
                    if (!(is_string(ctype) && ctype->as_string("type") == "function"))
                        bad("tools", "'" + cpath + "' must be a function tool (namespaces hold functions)");
                    const json::Value *cnm = c.find("name");
                    if (!is_string(cnm)) bad("tools", "'" + cpath + "' needs a string 'name'");
                    add_function_tool(c, ns, ns + "/" + cnm->as_string(cpath + ".name"), chat_tools,
                                      r.tool_registry, names, "tools");
                }
            } else if (tt == "custom") {
                // A Codex freeform tool (apply_patch and friends): the model writes raw text, not JSON arguments.
                // Internal form: a function tool with one required string parameter "input" - the output parser
                // keeps string-typed parameter values raw (bytes: newlines, Unicode, patch boundaries intact).
                const json::Value *nm = t.find("name");
                if (!is_string(nm)) bad("tools", "'" + path + "' needs a string 'name'");
                const std::string n = nm->as_string(path + ".name");
                check_tool_name(n, "tools");
                if (!names.insert(n).second) bad("tools", "duplicate tool name '" + n + "'");
                // The format rides into the parameter description (what the model reads): validated shape,
                // NOT enforced - this server has no Lark/regex runtime and never constrains the answer to it.
                std::string format_text;
                if (const json::Value *f = t.find("format")) {
                    if (!f->is_object()) bad("tools", "'" + path + ".format' must be an object");
                    const json::Value *ftype = f->find("type");
                    if (!(is_string(ftype) && ftype->as_string("type") == "grammar"))
                        bad("tools", "'" + path + ".format.type': only \"grammar\" is served, got '" +
                                (is_string(ftype) ? ftype->as_string("type") : std::string(json::type_name(ftype->type()))) + "'");
                    const json::Value *g = f->find("grammar");
                    if (!g || !g->is_object()) bad("tools", "'" + path + ".format' needs a grammar object");
                    const json::Value *gtype = g->find("type"), *def = g->find("definition");
                    if (!is_string(gtype)) bad("tools", "'" + path + ".format.grammar.type' must be a string");
                    if (!is_string(def) || def->as_string("definition").empty())
                        bad("tools", "'" + path + ".format.grammar.definition' must be a non-empty string");
                    format_text = "Raw input grammar (" + gtype->as_string("type") +
                                  ", shown to the model; the server does not enforce it):\n" +
                                  def->as_string("definition");
                }
                std::string desc;
                if (is_string(t.find("description"))) desc = t.find("description")->as_string(path + ".description");
                if (!desc.empty()) desc += "\n\n";
                desc += format_text.empty() ? "Input is raw text passed to the client verbatim." : format_text;
                json::Value props = json::Value::object();
                json::Value input = json::Value::object();
                input.set("type", json::Value::string("string"));
                input.set("description", json::Value::string(std::move(desc)));
                props.set("input", std::move(input));
                json::Value params = json::Value::object();
                params.set("type", json::Value::string("object"));
                params.set("properties", std::move(props));
                json::Value required = json::Value::array();
                required.push(json::Value::string("input"));
                params.set("required", std::move(required));
                json::Value fn = json::Value::object();
                fn.set("name", json::Value::string(n));
                fn.set("description", json::Value::string(format_text.empty() ? "Raw-text tool " + n
                                                                              : "Raw-text tool " + n + " (grammar shown in the input parameter)"));
                fn.set("parameters", std::move(params));
                json::Value entry = json::Value::object();
                entry.set("type", json::Value::string("function"));
                entry.set("function", std::move(fn));
                chat_tools.push(std::move(entry));
                json::Value reg = json::Value::object();
                reg.set("external", json::Value::string(n));
                reg.set("kind", json::Value::string("custom"));  // output: custom_tool_call, input byte-for-byte
                reg.set("namespace", json::Value::string(""));  // top-level (a custom tool is never namespaced)
                r.tool_registry.push(std::move(reg));
            } else if (tt == "web_search") {
                // A hosted tool: this server executes none (no OpenAI backend). Dropped from the generator's
                // view, disclosed in docs/server.md - the model can't call what it never sees, and the client
                // registry needs no entry. Installed Codex sends web_search by default, so refusing it would
                // make the route unusable, not more honest.
            } else {
                bad("tools", "tool type '" + tt + "' is not served (no hosted tools; functions and namespaces only)");
            }
        }
        if (chat_tools.as_array("tools").size() > 128) bad("tools", "at most 128 model-callable tools");
    }
    // tool_choice none: the generator sees no tools (calls can't be emitted); the registry stays (output mapping).
    r.chat.tools = (chat_tools.as_array("tools").empty() || tool_choice == "none") ? json::Value() : std::move(chat_tools);
}

}  // namespace

ResponsesRequest parse_responses_request(const json::Value &body, const SamplingParams &defaults,
                                         const std::string &served_model_id) {
    if (!body.is_object())
        bad("", "the request body must be a JSON object, got " + std::string(json::type_name(body.type())));
    ResponsesRequest r;
    r.chat.sampling = defaults;

    // Fields this build serves. Everything else is a 400 naming the key: an unknown field is a semantic the
    // server would silently ignore, and "no silent loss" beats protocol extensibility theater.
    static const std::set<std::string> kKnown = {
        "model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "stream", "store",
        "max_output_tokens", "temperature", "top_p", "seed", "stop", "reasoning", "include", "text", "truncation",
        "prompt_cache_key", "client_metadata", "metadata", "user", "safety_identifier"};
    for (const auto &kv : body.as_object("request"))
        if (!kKnown.count(kv.first))
            bad(kv.first, "'" + kv.first + "' is not served by this endpoint (see docs/server.md for the contract)");

    // model: the one truth. The Chat route serves any name and reports the real id; a Responses client keys
    // behavior off the name, so this route refuses to be anything other than what it is.
    const json::Value *model = field(body, "model");
    if (!is_string(model)) bad("model", "'model' must be a string naming the model to use");
    r.chat.model = model->as_string("model");
    if (r.chat.model != served_model_id)
        bad("model", "this server runs '" + served_model_id + "'; the request names '" + r.chat.model + "'");

    if (const json::Value *v = field(body, "stream")) r.chat.stream = get_bool(*v, "stream", "stream");
    if (const json::Value *v = field(body, "store")) {
        // Stateless by contract: no response is kept, so there is nothing a later call could store into.
        if (get_bool(*v, "store", "store")) bad("store", "'store': true is not served (stateless Responses; no "
                                                             "response store, no previous_response_id)");
    }
    if (const json::Value *v = field(body, "max_output_tokens")) {
        r.chat.max_tokens = get_int(*v, "max_output_tokens", "max_output_tokens");
        if (*r.chat.max_tokens < 1) bad("max_output_tokens", "'max_output_tokens' must be >= 1, got " +
                                                                 std::to_string(*r.chat.max_tokens));
    }
    if (const json::Value *v = field(body, "temperature")) {
        r.chat.sampling.temperature = get_number(*v, "temperature", "temperature");
        if (!(r.chat.sampling.temperature >= 0 && r.chat.sampling.temperature <= 2))
            bad("temperature", "'temperature' must be in [0, 2]");
    }
    if (const json::Value *v = field(body, "top_p")) {
        r.chat.sampling.top_p = get_number(*v, "top_p", "top_p");
        if (!(r.chat.sampling.top_p > 0 && r.chat.sampling.top_p <= 1)) bad("top_p", "'top_p' must be in (0, 1]");
    }
    if (const json::Value *v = field(body, "seed"))
        r.chat.sampling.seed = (uint64_t)get_int(*v, "seed", "seed"), r.chat.seeded = true;
    if (const json::Value *v = field(body, "stop")) {
        if (v->is_string()) r.chat.stop.push_back(v->as_string("stop"));
        else if (v->is_array())
            for (const json::Value &s : v->as_array("stop")) r.chat.stop.push_back(get_string(s, "stop", "stop"));
        else bad("stop", "'stop' must be a string or an array of strings");
        if (r.chat.stop.size() > 16) bad("stop", "at most 16 stop strings");
        for (const std::string &s : r.chat.stop)
            if (s.empty()) bad("stop", "stop strings must be non-empty");
    }

    // tool_choice: auto / none (the generator's tools on / off). A forced call is not served: the model's own
    // decision is the only one this engine parses, and "required" would ask the server to fake one.
    if (const json::Value *v = field(body, "tool_choice")) {
        if (!is_string(v)) bad("tool_choice", "'tool_choice' must be the string \"auto\" or \"none\"");
        r.tool_choice = v->as_string("tool_choice");
        if (r.tool_choice != "auto" && r.tool_choice != "none")
            bad("tool_choice", "'tool_choice': '" + r.tool_choice + "' is not served (only \"auto\" and \"none\")");
    }
    if (const json::Value *v = field(body, "parallel_tool_calls"); v) get_bool(*v, "parallel_tool_calls", "parallel_tool_calls");  // accepted: the parser emits every call the model writes

    // reasoning: effort maps onto the template's thinking block (the real knobs); a summary is never invented -
    // the raw reasoning streams, encrypted_content is reported null.
    if (const json::Value *v = field(body, "reasoning")) {
        if (!v->is_object()) bad("reasoning", "'reasoning' must be an object, got " + std::string(json::type_name(v->type())));
        for (const auto &kv : v->as_object("reasoning")) {
            const std::string &key = kv.first;
            if (key == "effort") {
                r.reasoning_effort = get_string(kv.second, "reasoning", "reasoning.effort");
                const std::string &e = r.reasoning_effort;
                if (e == "none") r.chat.template_options.enable_thinking = false;
                else if (e == "minimal" || e == "low") r.chat.template_options.reasoning_effort = "low";
                else if (e == "medium") r.chat.template_options.reasoning_effort = "medium";
                else if (e == "high" || e == "xhigh") r.chat.template_options.reasoning_effort = "xhigh";
                else bad("reasoning", "'reasoning.effort' must be none, minimal, low, medium, high, or xhigh, got '" +
                                          e + "'");
            } else if (key == "summary") {
                get_string(kv.second, "reasoning", "reasoning.summary");  // accepted; raw reasoning is what streams
            } else {
                bad("reasoning", "'reasoning." + key + "' is not served");
            }
        }
    }
    if (const json::Value *v = field(body, "include")) {
        if (!v->is_array()) bad("include", "'include' must be an array of response-inclusion strings");
        for (const json::Value &inc : v->as_array("include")) {
            const std::string s = get_string(inc, "include", "include[]");
            // The only inclusion installed Codex asks for. Answered honestly: reasoning streams raw and
            // encrypted_content is always null (this server has no key to encrypt reasoning with).
            if (s != "reasoning.encrypted_content")
                bad("include", "'include': '" + s + "' is not served (raw reasoning streams; encrypted_content is null)");
        }
    }
    if (const json::Value *v = field(body, "text")) {
        // The structured-output surface joins grammar enforcement at its story (docs/server.md); plain text is
        // what this build produces, so anything else is a named refusal, never a silent downgrade.
        const json::Value *fmt = v->is_object() ? v->find("format") : nullptr;
        if (!fmt || !fmt->is_object()) bad("text", "'text' must be an object with a 'format' object");
        const json::Value *ftype = fmt->find("type");
        if (!(is_string(ftype) && ftype->as_string("type") == "text"))
            bad("text", "'text.format.type': only \"text\" is served (structured output: docs/server.md)");
    }
    if (const json::Value *v = field(body, "truncation")) {
        const std::string t = get_string(*v, "truncation", "truncation");
        if (t != "disabled")
            bad("truncation", "'truncation': \"" + t + "\" is not served - a prompt over the context is refused, "
                              "not truncated (\"disabled\" is the only mode that is true of this server)");
    }
    // Benign client metadata: validated in shape, kept out of generation.
    if (const json::Value *v = field(body, "prompt_cache_key")) {
        const std::string k = get_string(*v, "prompt_cache_key", "prompt_cache_key");
        if (k.size() > 512) bad("prompt_cache_key", "'prompt_cache_key' is longer than 512 bytes");
    }
    for (const char *meta : {"client_metadata", "metadata"}) {
        if (const json::Value *v = field(body, meta)) {
            if (!v->is_object()) bad(meta, "'" + std::string(meta) + "' must be an object of string values");
            const auto &obj = v->as_object(meta);
            if (obj.size() > 64) bad(meta, "'" + std::string(meta) + "' has more than 64 entries");
            for (const auto &kv : obj) {
                get_string(kv.second, meta, std::string(meta) + "." + kv.first);
                if (kv.first.size() > 256) bad(meta, "'" + std::string(meta) + "' has a key longer than 256 bytes");
            }
        }
    }
    for (const char *skey : {"user", "safety_identifier"})
        if (const json::Value *v = field(body, skey)) get_string(*v, skey, skey);

    // input (+ instructions): the conversation, in order. instructions is a leading system message; items keep
    // their order; the template needs at least one user message (as the Chat route requires).
    json::Value messages = json::Value::array();
    bool has_user = false;
    if (const json::Value *v = field(body, "instructions")) {
        const std::string text = get_string(*v, "instructions", "instructions");
        if (!text.empty()) {
            json::Value m = json::Value::object();
            m.set("role", json::Value::string("system"));
            m.set("content", json::Value::string(std::move(text)));
            messages.push(std::move(m));
        }
    }
    const json::Value *input = field(body, "input");
    if (!input) bad("input", "'input' is required (a string or an array of input items)");
    if (input->is_string()) {
        if (input->as_string("input").empty()) bad("input", "'input' must not be an empty string");
        json::Value m = json::Value::object();
        m.set("role", json::Value::string("user"));
        m.set("content", json::Value::string(input->as_string("input")));
        messages.push(std::move(m));
        has_user = true;
    } else if (input->is_array()) {
        const auto &items = input->as_array("input");
        if (items.empty()) bad("input", "'input' must be a non-empty array of items");
        for (size_t k = 0; k < items.size(); ++k) {
            json::Value m = normalize_item(items[k], k);
            if (m.find("role")->as_string("role") == "user") has_user = true;
            messages.push(std::move(m));
        }
    } else {
        bad("input", "'input' must be a string or an array of input items, got " +
                         std::string(json::type_name(input->type())));
    }
    if (!has_user) bad("input", "the input has no user message (the template requires a query to answer)");
    r.chat.messages = std::move(messages);

    parse_tools(field(body, "tools"), r, r.tool_choice);
    return r;
}

}  // namespace strix
