#!/usr/bin/env node
// Responses-API smoke against a RUNNING drinkme server, driven by the
// OFFICIAL openai SDK — the client the Agents SDK and Codex are built on.
// Does `client.responses.create` (non-stream and stream, with and without a
// function tool) and a function_call_output round trip land, and does the
// SDK's own stream accumulator accept our event sequence?
//
//   cd bench/.node && npm i            # or: bun install — once; node_modules is ignored
//   node bench/responses_sdk_smoke.mjs --base-url http://127.0.0.1:3216/v1 \
//        -o measurements/responses_sdk_smoke_<model>.json
//
// Verdicts printed, one per probe, every raw response kept in the receipt;
// exit 0 when every probe holds, 1 otherwise, 2 when the SDK is not
// installed. A verdict is a claim about THIS server on THIS model. Against
// bench/fake_server.py (CPU, canned replies) ALL HOLD means the wire holds;
// against a real bottle it means the model held too.
//
// The prompts and verdict rules mirror bench/serve_dialect_smoke.py's so the
// two receipts read side by side: 17*23 -> "391" with no leaked reasoning
// or tool markup in the text; the weather-in-Hilo tool call parsed with
// city=hilo; the result fed back and used (27 / rain) without a re-call.

import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const here = dirname(fileURLToPath(import.meta.url));

async function loadSDK() {
  // the scratch install under bench/.node first (what the header says to
  // do), then whatever `openai` resolves to from here
  for (const from of [join(here, ".node", "package.json"), join(here, "package.json")]) {
    try {
      return createRequire(from)("openai");
    } catch (e) {
      if (e.code !== "MODULE_NOT_FOUND") throw e;
    }
  }
  try {
    return (await import("openai")).default;
  } catch (e) {
    if (e.code !== "ERR_MODULE_NOT_FOUND") throw e;
  }
  return null;
}

const { values: args } = parseArgs({
  options: {
    "base-url": { type: "string", default: "http://127.0.0.1:3215/v1" },
    model: { type: "string" },
    "max-tokens": { type: "string", default: "1500" },  // a thinking model spends most of it reasoning
    timeout: { type: "string", default: "900" },
    out: { type: "string", short: "o" },
    only: { type: "string" },
  },
});

const WEATHER_TOOL = {
  type: "function",
  name: "get_weather",
  description: "Current weather for a city.",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: "City name" },
      unit: { type: "string", enum: ["c", "f"] },
    },
    required: ["city"],
  },
  strict: false,
};

// Any of these in visible text means a reasoning or tool channel leaked.
const LEAK_MARKERS = [
  "<think>", "</think>", "<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>",
  "<tool_call>", "<function=", "<start_function_call>", "<|tool>", "[TOOL_CALLS]",
  "<minimax:tool_call>", "<arg_key>",
];
const leaks = (text) => LEAK_MARKERS.filter((m) => text && text.includes(m));

const textOf = (resp) =>
  resp.output
    .filter((i) => i.type === "message")
    .flatMap((i) => i.content)
    .filter((c) => c.type === "output_text")
    .map((c) => c.text)
    .join("");
const callsOf = (resp) => resp.output.filter((i) => i.type === "function_call");
const reasoningOf = (resp) => resp.output.filter((i) => i.type === "reasoning");
const parseArgsOf = (call) => {
  try {
    return JSON.parse(call.arguments);
  } catch {
    return null;
  }
};

async function main() {
  const OpenAI = await loadSDK();
  if (!OpenAI) {
    console.error("[smoke] the openai SDK is not installed. Run once:\n" +
      `  cd ${resolve(here, ".node")} && npm i     # or: bun install`);
    return 2;
  }
  const base = args["base-url"].replace(/\/+$/, "");
  const maxTokens = parseInt(args["max-tokens"], 10);
  const timeoutMs = parseFloat(args.timeout) * 1000;
  const only = args.only ? new Set(args.only.split(",")) : null;
  const want = (name) => only === null || only.has(name);
  const client = new OpenAI({ baseURL: base, apiKey: "x", timeout: timeoutMs, maxRetries: 0 });

  const receipt = { base_url: base, sdk: "openai (node)", started: new Date().toISOString(), probes: {} };
  const verdicts = {};
  const say = (line) => console.log(`[smoke] ${line}`);
  const mark = (ok) => (ok ? "OK" : "FAIL");

  const models = await client.models.list();
  receipt.models = models.data;
  const model = args.model || models.data[0].id;
  const announced = models.data.find((m) => m.id === model) || models.data[0];
  say(`model ${model} · announced: thinking=${announced.thinking} tool_format=${announced.tool_format}`);

  const plainInput = "What is 17 * 23? Reply with just the number.";

  // ---- probe 1: plain, non-stream --------------------------------------
  if (want("plain")) {
    let resp, err = null;
    try {
      resp = await client.responses.create({ model, input: plainInput, max_output_tokens: maxTokens });
    } catch (e) { err = String(e); }
    const text = resp ? textOf(resp) : "";
    const lk = leaks(text);
    const ok = !!resp && resp.status === "completed" && !lk.length && text.includes("391");
    verdicts.plain = ok;
    receipt.probes.plain = { error: err, response: resp, leaks: lk };
    say(`plain: ${err ? "ERROR " + err : resp.status} · leaks=${lk.length ? lk : "none"} · ` +
      `reasoning_item=${resp && reasoningOf(resp).length ? "yes" : "no"} · answer_present=${text.includes("391")} → ${mark(ok)}`);
  }

  // ---- probe 2: plain, streamed through the SDK's accumulator ------------
  if (want("plain_stream")) {
    let final, err = null;
    const deltas = [];
    const types = [];
    try {
      const stream = client.responses.stream({ model, input: plainInput, max_output_tokens: maxTokens });
      for await (const ev of stream) {
        types.push(ev.type);
        if (ev.type === "response.output_text.delta") deltas.push(ev.delta);
      }
      final = await stream.finalResponse();  // throws if the events did not line up
    } catch (e) { err = String(e); }
    const text = final ? textOf(final) : "";
    const lk = leaks(text);
    const identity = !!final && deltas.join("") === text;
    const ok = !!final && final.status === "completed" && !lk.length && text.includes("391") && identity &&
      types[0] === "response.created" && types[types.length - 1] === "response.completed";
    verdicts.plain_stream = ok;
    receipt.probes.plain_stream = { error: err, response: final, event_types: types, n_events: types.length, leaks: lk, delta_identity: identity };
    say(`plain/stream: ${err ? "ERROR " + err : final.status} · ${types.length} events · deltas==final_text=${identity} · ` +
      `leaks=${lk.length ? lk : "none"} → ${mark(ok)}`);
  }

  const toolInput = [{ role: "user", content: "What's the weather in Hilo right now? Use the tool." }];
  let toolResp = null;

  // ---- probe 3: tool call OUT, non-stream --------------------------------
  if (want("tool_out")) {
    let resp, err = null;
    try {
      resp = await client.responses.create({ model, input: toolInput, tools: [WEATHER_TOOL], max_output_tokens: maxTokens });
    } catch (e) { err = String(e); }
    const calls = resp ? callsOf(resp) : [];
    const parsed = calls.length > 0 && calls[0].name === "get_weather";
    const a = parsed ? parseArgsOf(calls[0]) : null;
    const argok = !!a && typeof a.city === "string" && a.city.toLowerCase().includes("hilo");
    const lk = leaks(resp ? textOf(resp) : "");
    const ok = !!resp && resp.status === "completed" && parsed && argok && !lk.length &&
      typeof calls[0].call_id === "string" && calls[0].call_id.length > 0;
    verdicts.tool_out = ok;
    receipt.probes.tool_out = { error: err, response: resp, leaks: lk };
    if (ok) toolResp = resp;
    say(`tool out: ${err ? "ERROR " + err : resp.status} · parsed=${parsed} args_ok=${argok} leaks=${lk.length ? lk : "none"} → ${mark(ok)}`);
  }

  // ---- probe 4: tool call OUT, streamed ----------------------------------
  if (want("tool_out_stream")) {
    let final, err = null;
    const argDeltas = [];
    const types = [];
    try {
      const stream = client.responses.stream({ model, input: toolInput, tools: [WEATHER_TOOL], max_output_tokens: maxTokens });
      for await (const ev of stream) {
        types.push(ev.type);
        if (ev.type === "response.function_call_arguments.delta") argDeltas.push(ev.delta);
      }
      final = await stream.finalResponse();
    } catch (e) { err = String(e); }
    const calls = final ? callsOf(final) : [];
    const parsed = calls.length > 0 && calls[0].name === "get_weather";
    const identity = parsed && argDeltas.join("") === calls[0].arguments;
    const lk = leaks(final ? textOf(final) : "");
    const ok = !!final && parsed && identity && !lk.length &&
      types.includes("response.function_call_arguments.done") && types.includes("response.output_item.done");
    verdicts.tool_out_stream = ok;
    receipt.probes.tool_out_stream = { error: err, response: final, event_types: types, leaks: lk, arguments_identity: identity };
    say(`tool out/stream: ${err ? "ERROR " + err : final.status} · parsed=${parsed} deltas==arguments=${identity} leaks=${lk.length ? lk : "none"} → ${mark(ok)}`);
  }

  // ---- probe 5: function_call_output round trip -------------------------
  if (want("tool_in")) {
    if (toolResp) {
      const call = callsOf(toolResp)[0];
      let resp, err = null;
      try {
        resp = await client.responses.create({
          model, tools: [WEATHER_TOOL], max_output_tokens: maxTokens,
          input: [
            ...toolInput,
            ...toolResp.output,  // the items exactly as they came back (reasoning, message, function_call)
            { type: "function_call_output", call_id: call.call_id,
              output: JSON.stringify({ city: "Hilo", temp_c: 27, sky: "light rain" }) },
          ],
        });
      } catch (e) { err = String(e); }
      const text = resp ? textOf(resp) : "";
      const used = /27|rain/i.test(text);
      const recalled = !!resp && callsOf(resp).length > 0;
      const lk = leaks(text);
      const ok = !!resp && resp.status === "completed" && used && !recalled && !lk.length;
      verdicts.tool_in = ok;
      receipt.probes.tool_in = { error: err, response: resp, leaks: lk };
      say(`tool in: ${err ? "ERROR " + err : resp.status} · used_result=${used} re-called=${recalled} leaks=${lk.length ? lk : "none"} → ${mark(ok)}`);
    } else {
      verdicts.tool_in = false;
      say("tool in: skipped (no parsed call to feed back)");
    }
  }

  receipt.verdicts = verdicts;
  receipt.finished = new Date().toISOString();
  if (args.out) {
    mkdirSync(dirname(resolve(args.out)), { recursive: true });
    writeFileSync(args.out, JSON.stringify(receipt, null, 1));
    say(`receipt → ${args.out}`);
  }
  const failed = Object.entries(verdicts).filter(([, v]) => !v).map(([k]) => k);
  say(failed.length ? `FAILED: ${failed.join(", ")}` : "ALL HOLD");
  return failed.length ? 1 : 0;
}

main().then((rc) => process.exit(rc), (e) => {
  console.error(`[smoke] ${e && e.stack ? e.stack : e}`);
  process.exit(1);
});
